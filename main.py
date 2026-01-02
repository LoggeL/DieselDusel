import os
import logging
import json
import base64
import sqlite3
from io import BytesIO
from typing import Optional
from datetime import datetime

from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)
from pydantic import BaseModel, Field
import requests
from dotenv import load_dotenv
from PIL import Image

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states for /modify command
MODIFY_SELECT_FIELD, MODIFY_ENTER_VALUE = range(2)

# Database path - use /app/data in container, local directory otherwise
DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(DATA_DIR, "dieseldusel.db")


class ImageClassification(BaseModel):
    """Image type classification"""

    description: str = Field(
        ...,
        description="A detailed description of the image that was used for the classification",
    )
    receipt: bool = Field(...)
    dashboard: bool = Field(...)


class DashboardStats(BaseModel):
    """Dashboard Stats from car display"""

    consumption: Optional[float] = Field(
        default=None,
        description="fuel consumption noted with 'Verbrauch' in the unit l/100km as stated on the car's dashboard",
    )
    total_km: Optional[int] = Field(
        default=None,
        description="total Kilometer noted with 'km' as stated on the car's dashboard",
    )
    trip_km: Optional[float] = Field(
        default=None,
        description="Trip km (number on bottom right of the display) in kilometers as stated on the car's dashboard",
    )


class ReceiptStats(BaseModel):
    """Fuel station receipt stats"""

    price_per_liter: Optional[float] = Field(
        default=None,
        description="price per liter (in Euro) as stated on the receipt (e.g. 1.599)",
    )
    total_cost: Optional[float] = Field(
        default=None, description="total cost in Euro from the receipt"
    )
    date: Optional[str] = Field(
        default=None, description="date of the receipt in the format YYYY-MM-DD"
    )


def init_database():
    """Initialize the SQLite database with the fuel_logs table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fuel_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT,
            total_km INTEGER,
            trip_km REAL,
            liters REAL,
            costs REAL,
            euro_per_liter REAL,
            consumption REAL,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON fuel_logs(user_id)")
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


class TelegramBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")

        # Initialize clients
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_api_base = "https://openrouter.ai/api/v1"
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        logger.info("Clients initialized")

        if not self.telegram_token or not self.openrouter_api_key:
            logger.error("Missing required environment variables")
            raise ValueError("Missing required environment variables")

        # Initialize database
        init_database()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        logger.info(f"Start command received from user {update.effective_user.id}")
        welcome_message = (
            "Welcome to the Fuel Stats Bot!\n\n"
            "Send me a photo of your car's display, and I'll extract the following information:\n"
            "- Fuel consumption (l/100km)\n"
            "- Total kilometers\n"
            "- Trip kilometers\n\n"
            "After scanning both dashboard and receipt, use /save to store your data.\n\n"
            "Use /help to see all available commands."
        )
        await update.message.reply_text(welcome_message)
        logger.info(f"Welcome message sent to user {update.effective_user.id}")

    async def help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Send a message when the command /help is issued."""
        logger.info(f"Help command received from user {update.effective_user.id}")
        help_text = (
            "*Available Commands:*\n\n"
            "*Scanning:*\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/cancel - Cancel current session\n\n"
            "*Data Management:*\n"
            "/save - Save last scan to database\n"
            "/history - View recent fuel logs\n"
            "/stats - View fuel statistics\n"
            "/export - Export all data as CSV\n"
            "/modify - Edit a log entry\n"
            "/delete - Delete a log entry\n"
            "/clear - Clear all your data\n\n"
            "*How to use:*\n"
            "1. Send a photo of your car's dashboard\n"
            "2. Send a photo of your fuel receipt\n"
            "3. Use /save to store the data"
        )
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"Help message sent to user {update.effective_user.id}")

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Cancel the current image processing session."""
        user_id = update.effective_user.id
        logger.info(f"Cancel command received from user {user_id}")

        if "previous_stats" in context.user_data:
            context.user_data["previous_stats"] = None
            context.user_data["last_complete_scan"] = None
            await update.message.reply_text(
                "Session cancelled. You can now start over by sending a new image."
            )
        else:
            await update.message.reply_text(
                "No active session to cancel. You can start by sending an image."
            )
        logger.info(f"Session cancelled for user {user_id}")

    async def save(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Save the last complete scan to the database."""
        user_id = update.effective_user.id
        logger.info(f"Save command received from user {user_id}")

        last_scan = context.user_data.get("last_complete_scan")
        if not last_scan:
            await update.message.reply_text(
                "No scan to save. Please send a dashboard and receipt photo first."
            )
            return

        dashboard_stats = last_scan.get("dashboard")
        receipt_stats = last_scan.get("receipt")
        note = last_scan.get("note", "")

        liters = (
            receipt_stats.total_cost / receipt_stats.price_per_liter
            if receipt_stats.price_per_liter and receipt_stats.total_cost
            else None
        )

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO fuel_logs (user_id, date, total_km, trip_km, liters, costs, euro_per_liter, consumption, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                receipt_stats.date,
                dashboard_stats.total_km,
                dashboard_stats.trip_km,
                liters,
                receipt_stats.total_cost,
                receipt_stats.price_per_liter,
                dashboard_stats.consumption,
                note,
            ),
        )
        conn.commit()
        log_id = cursor.lastrowid
        conn.close()

        context.user_data["last_complete_scan"] = None
        await update.message.reply_text(
            f"Saved! Entry #{log_id} added to your fuel log."
        )
        logger.info(f"Saved fuel log #{log_id} for user {user_id}")

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show recent fuel logs for the user."""
        user_id = update.effective_user.id
        logger.info(f"History command received from user {user_id}")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, date, total_km, trip_km, liters, costs, consumption
            FROM fuel_logs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        """,
            (user_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            await update.message.reply_text(
                "No fuel logs found. Send photos to start tracking!"
            )
            return

        lines = ["*Recent Fuel Logs:*\n"]
        for row in rows:
            log_id, date, total_km, trip_km, liters, costs, consumption = row
            date_str = date or "N/A"
            km_str = f"{total_km:,}" if total_km else "N/A"
            trip_str = f"{trip_km:.1f}" if trip_km else "N/A"
            liters_str = f"{liters:.2f}L" if liters else "N/A"
            costs_str = f"{costs:.2f}EUR" if costs else "N/A"
            cons_str = f"{consumption:.1f}" if consumption else "N/A"

            lines.append(
                f"*#{log_id}* | {date_str}\n"
                f"  {km_str} km | {trip_str} km\n"
                f"  {liters_str} | {costs_str} | {cons_str} l/100km\n"
            )

        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show fuel statistics for the user."""
        user_id = update.effective_user.id
        logger.info(f"Stats command received from user {user_id}")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) as count,
                SUM(liters) as total_liters,
                SUM(costs) as total_costs,
                SUM(trip_km) as total_trip_km,
                AVG(consumption) as avg_consumption,
                AVG(euro_per_liter) as avg_price,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM fuel_logs
            WHERE user_id = ?
        """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()

        count = row[0] or 0
        if count == 0:
            await update.message.reply_text(
                "No fuel logs found. Send photos to start tracking!"
            )
            return

        total_liters = row[1] or 0
        total_costs = row[2] or 0
        total_trip_km = row[3] or 0
        avg_consumption = row[4] or 0
        avg_price = row[5] or 0
        first_date = row[6] or "N/A"
        last_date = row[7] or "N/A"

        stats_text = (
            "*Your Fuel Statistics:*\n\n"
            f"Total entries: {count}\n"
            f"Period: {first_date} - {last_date}\n\n"
            f"Total fuel: {total_liters:.2f} L\n"
            f"Total spent: {total_costs:.2f} EUR\n"
            f"Total distance: {total_trip_km:.1f} km\n\n"
            f"Avg consumption: {avg_consumption:.2f} l/100km\n"
            f"Avg price: {avg_price:.3f} EUR/L"
        )

        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)

    async def export(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Export all user data as CSV."""
        user_id = update.effective_user.id
        logger.info(f"Export command received from user {user_id}")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT date, total_km, trip_km, liters, costs, euro_per_liter, consumption, note
            FROM fuel_logs
            WHERE user_id = ?
            ORDER BY date ASC
        """,
            (user_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            await update.message.reply_text("No fuel logs found. Nothing to export!")
            return

        csv_lines = [
            "date;total-km;trip-km;liter;costs;euro_per_liter;consumption;note"
        ]
        for row in rows:
            csv_lines.append(";".join(str(v) if v is not None else "" for v in row))

        csv_content = "\n".join(csv_lines)
        csv_file = BytesIO(csv_content.encode("utf-8"))
        csv_file.name = f"fuel_logs_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv"

        await update.message.reply_document(
            document=csv_file,
            caption=f"Exported {len(rows)} fuel log entries.",
            filename=csv_file.name,
        )

    async def delete(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Delete a specific log entry."""
        user_id = update.effective_user.id
        logger.info(f"Delete command received from user {user_id}")

        args = context.args
        if not args:
            await update.message.reply_text(
                "Usage: `/delete <id>`\n\nExample: `/delete 5`\n\nUse /history to see entry IDs.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        try:
            log_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Invalid ID. Please provide a number.")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM fuel_logs WHERE id = ? AND user_id = ?", (log_id, user_id)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted:
            await update.message.reply_text(f"Entry #{log_id} deleted.")
        else:
            await update.message.reply_text(
                f"Entry #{log_id} not found or doesn't belong to you."
            )

    async def clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Clear all user data (with confirmation)."""
        user_id = update.effective_user.id
        logger.info(f"Clear command received from user {user_id}")

        args = context.args
        if not args or args[0].lower() != "confirm":
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM fuel_logs WHERE user_id = ?", (user_id,)
            )
            count = cursor.fetchone()[0]
            conn.close()

            await update.message.reply_text(
                f"This will delete all {count} of your fuel log entries.\n\n"
                "To confirm, type: `/clear confirm`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fuel_logs WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        await update.message.reply_text(
            f"Deleted {deleted} entries. Your data has been cleared."
        )

    async def modify_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Start the modify conversation."""
        user_id = update.effective_user.id
        logger.info(f"Modify command received from user {user_id}")

        args = context.args
        if not args:
            await update.message.reply_text(
                "Usage: `/modify <id>`\n\nExample: `/modify 5`\n\nUse /history to see entry IDs.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return ConversationHandler.END

        try:
            log_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Invalid ID. Please provide a number.")
            return ConversationHandler.END

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, date, total_km, trip_km, liters, costs, euro_per_liter, consumption, note
            FROM fuel_logs
            WHERE id = ? AND user_id = ?
        """,
            (log_id, user_id),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            await update.message.reply_text(
                f"Entry #{log_id} not found or doesn't belong to you."
            )
            return ConversationHandler.END

        context.user_data["modify_entry"] = {
            "id": row[0],
            "date": row[1],
            "total_km": row[2],
            "trip_km": row[3],
            "liters": row[4],
            "costs": row[5],
            "euro_per_liter": row[6],
            "consumption": row[7],
            "note": row[8],
        }

        fields = [
            ("1", "Date", row[1]),
            ("2", "Total KM", row[2]),
            ("3", "Trip KM", row[3]),
            ("4", "Liters", row[4]),
            ("5", "Costs (EUR)", row[5]),
            ("6", "Price/L (EUR)", row[6]),
            ("7", "Consumption (l/100km)", row[7]),
            ("8", "Note", row[8]),
        ]

        lines = [f"*Editing entry #{log_id}*\n"]
        for num, name, val in fields:
            display_val = val if val is not None else "(empty)"
            lines.append(f"*{num}.* {name}: `{display_val}`")

        lines.append("\nReply with the field number to edit, or /done to finish.")

        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        return MODIFY_SELECT_FIELD

    async def modify_select_field(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle field selection for modify."""
        text = update.message.text.strip()

        field_map = {
            "1": ("date", "Date (YYYY-MM-DD)"),
            "2": ("total_km", "Total KM"),
            "3": ("trip_km", "Trip KM"),
            "4": ("liters", "Liters"),
            "5": ("costs", "Costs (EUR)"),
            "6": ("euro_per_liter", "Price per Liter (EUR)"),
            "7": ("consumption", "Consumption (l/100km)"),
            "8": ("note", "Note"),
        }

        if text not in field_map:
            await update.message.reply_text(
                "Invalid selection. Enter a number 1-8, or /done to finish."
            )
            return MODIFY_SELECT_FIELD

        field_key, field_name = field_map[text]
        context.user_data["modify_field"] = field_key
        context.user_data["modify_field_name"] = field_name

        current_val = context.user_data["modify_entry"].get(field_key)
        display_val = current_val if current_val is not None else "(empty)"

        await update.message.reply_text(
            f"Current value for *{field_name}*: `{display_val}`\n\nEnter the new value:",
            parse_mode=ParseMode.MARKDOWN,
        )
        return MODIFY_ENTER_VALUE

    async def modify_enter_value(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Handle new value entry for modify."""
        text = update.message.text.strip()
        field_key = context.user_data["modify_field"]
        field_name = context.user_data["modify_field_name"]

        # Parse value based on field type
        new_value = text
        if field_key in ("total_km",):
            try:
                new_value = int(text) if text else None
            except ValueError:
                await update.message.reply_text("Please enter a valid integer.")
                return MODIFY_ENTER_VALUE
        elif field_key in (
            "trip_km",
            "liters",
            "costs",
            "euro_per_liter",
            "consumption",
        ):
            try:
                new_value = float(text) if text else None
            except ValueError:
                await update.message.reply_text("Please enter a valid number.")
                return MODIFY_ENTER_VALUE

        # Update in memory
        context.user_data["modify_entry"][field_key] = new_value

        # Update in database
        entry = context.user_data["modify_entry"]
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE fuel_logs SET {field_key} = ? WHERE id = ?",
            (new_value, entry["id"]),
        )
        conn.commit()
        conn.close()

        await update.message.reply_text(
            f"Updated *{field_name}* to `{new_value}`\n\n"
            "Enter another field number to edit, or /done to finish.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return MODIFY_SELECT_FIELD

    async def modify_done(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Finish the modify conversation."""
        entry = context.user_data.get("modify_entry", {})
        entry_id = entry.get("id", "?")

        context.user_data.pop("modify_entry", None)
        context.user_data.pop("modify_field", None)
        context.user_data.pop("modify_field_name", None)

        await update.message.reply_text(f"Finished editing entry #{entry_id}.")
        return ConversationHandler.END

    async def modify_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        """Cancel the modify conversation."""
        context.user_data.pop("modify_entry", None)
        context.user_data.pop("modify_field", None)
        context.user_data.pop("modify_field_name", None)

        await update.message.reply_text("Edit cancelled.")
        return ConversationHandler.END

    def encode_image(self, image: bytes) -> str:
        """Encode the image to base64."""
        logger.debug("Encoding image to base64")
        return base64.b64encode(image).decode("utf-8")

    def process_image(self, image_bytes: bytes) -> Optional[tuple[str, BaseModel]]:
        """Process the image using OpenRouter API and return classification and stats."""
        logger.info("Starting image processing")
        try:
            # Convert to JPEG and optimize
            image = Image.open(BytesIO(image_bytes))
            output_buffer = BytesIO()
            image.save(output_buffer, format="JPEG", quality=85, optimize=True)
            logger.debug("Image converted and optimized")
            base64_image = self.encode_image(output_buffer.getvalue())
            logger.debug("Image encoded")

            # Create OpenRouter API request for both classification and processing
            logger.info("Sending processing request to OpenRouter API")
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/LoggeL/DieselDusel",
                "Content-Type": "application/json",
                "X-Title": "DieselDusel Bot",
            }

            payload = {
                "model": "google/gemini-2.5-flash-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Analyze this image and provide two things:
1. Classification: Determine if this is a dashboard or receipt image according to this schema: {ImageClassification.model_json_schema()}
2. Data Extraction: Extract the relevant data according to this schema: {DashboardStats.model_json_schema()} for dashboard or {ReceiptStats.model_json_schema()} for receipt

Provide the response as a JSON object with two fields:
- classification: The classification result
- data: The extracted data based on the classification""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1024,
                "top_p": 1,
                "response_format": {"type": "json_object"},
            }

            # Make sure we're using the correct OpenRouter endpoint
            url = f"{self.openrouter_api_base}/chat/completions"
            logger.debug(f"Sending request to: {url}")

            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
                return None

            response_data = response.json()
            response_text = response_data["choices"][0]["message"]["content"]
            logger.debug(f"OpenRouter API response: {response_text}")
            print(response_text)

            # Parse the combined response
            response_data = json.loads(response_text)
            classification = ImageClassification.model_validate(
                response_data["classification"]
            )
            classification_type = "dashboard" if classification.dashboard else "receipt"

            # Parse the data based on classification type
            model_class = (
                DashboardStats if classification_type == "dashboard" else ReceiptStats
            )
            stats = model_class.model_validate(response_data["data"])

            logger.info(
                f"Image processed as: {classification_type} with stats: {stats}"
            )
            return classification_type, stats

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None

    def format_dashboard_stats(self, stats: DashboardStats) -> str:
        """Format dashboard statistics for display."""
        consumption = (
            f"{stats.consumption:.1f}" if stats.consumption is not None else "Unknown"
        )
        total_km = f"{stats.total_km:,}" if stats.total_km is not None else "Unknown"
        trip_km = f"{stats.trip_km:.1f}" if stats.trip_km is not None else "Unknown"

        return (
            f"Consumption: {consumption} l/100km\n"
            f"Total Distance: {total_km} km\n"
            f"Trip Distance: {trip_km} km"
        )

    def format_receipt_stats(self, stats: ReceiptStats) -> str:
        """Format receipt statistics for display."""
        date = stats.date if stats.date else "Unknown"
        price = (
            f"{stats.price_per_liter:.3f}"
            if stats.price_per_liter is not None
            else "Unknown"
        )
        total = f"{stats.total_cost:.2f}" if stats.total_cost is not None else "Unknown"

        return f"Date: {date}\nPrice: {price} EUR/l\nTotal: {total} EUR"

    def create_csv_file(
        self, dashboard_stats: DashboardStats, receipt_stats: ReceiptStats, note: str
    ) -> BytesIO:
        """Create a CSV file in memory."""
        liters = (
            receipt_stats.total_cost / receipt_stats.price_per_liter
            if receipt_stats.price_per_liter and receipt_stats.total_cost
            else 0
        )

        csv_content = (
            "date;total-km;trip-km;liter;costs;euro_per_liter;consumption;note\n"
            f"{receipt_stats.date or ''};{dashboard_stats.total_km or ''};"
            f"{dashboard_stats.trip_km or ''};{liters:.2f};"
            f"{receipt_stats.total_cost or ''};{receipt_stats.price_per_liter or ''};"
            f"{dashboard_stats.consumption or ''};{note}\n"
        )

        output = BytesIO(csv_content.encode("utf-8"))
        output.name = f"fuel_log_{receipt_stats.date or 'unknown'}.csv"
        return output

    async def handle_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming photos."""
        user_id = update.effective_user.id
        logger.info(f"Received photo from user {user_id}")

        # Send typing action
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        try:
            # Send initial processing message
            processing_message = await update.message.reply_text(
                "Processing your image... Please wait."
            )
            logger.debug("Sent processing message")

            # If the message has caption, add it as a note
            note = update.message.caption if update.message.caption else ""

            # Get the photo file
            photo = await update.message.photo[-1].get_file()
            image_bytes = await photo.download_as_bytearray()
            logger.debug("Photo downloaded")

            # Process the image
            result = self.process_image(image_bytes)
            if not result:
                logger.warning(f"Image processing failed for user {user_id}")
                await processing_message.edit_text(
                    "Could not process the image. Please try again with a clearer photo."
                )
                return

            image_type, stats = result
            logger.info(f"Image processed with type {image_type} for user {user_id}")

            if not context.user_data.get("previous_stats"):
                # Store first image stats
                context.user_data["previous_stats"] = stats
                logger.info(f"Stored first image stats for user {user_id}")

                formatted_stats = (
                    self.format_dashboard_stats(stats)
                    if image_type == "dashboard"
                    else self.format_receipt_stats(stats)
                )

                next_type = "receipt" if image_type == "dashboard" else "dashboard"

                await processing_message.edit_text(
                    f"*{image_type.title()} Scanned*\n\n"
                    f"{formatted_stats}\n\n"
                    f"Please send the {next_type} image next.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # We have both images now
            previous_stats = context.user_data["previous_stats"]
            context.user_data["previous_stats"] = None  # Clear stored stats
            logger.info(f"Processing second image for user {user_id}")

            # Check if we have one of each type
            if (
                isinstance(previous_stats, DashboardStats)
                and isinstance(stats, DashboardStats)
            ) or (
                isinstance(previous_stats, ReceiptStats)
                and isinstance(stats, ReceiptStats)
            ):
                logger.warning(f"Duplicate image types received from user {user_id}")
                await processing_message.edit_text(
                    "Both images appear to be the same type. Please send one dashboard image and one receipt image."
                )
                return

            # Determine which is which
            dashboard_stats = (
                previous_stats if isinstance(previous_stats, DashboardStats) else stats
            )
            receipt_stats = (
                previous_stats if isinstance(previous_stats, ReceiptStats) else stats
            )
            logger.debug(f"Dashboard stats: {dashboard_stats}")
            logger.debug(f"Receipt stats: {receipt_stats}")

            # Store the complete scan for /save command
            context.user_data["last_complete_scan"] = {
                "dashboard": dashboard_stats,
                "receipt": receipt_stats,
                "note": note,
            }

            # Format the merged response message
            response = (
                "*Analysis Complete!*\n\n"
                f"{self.format_dashboard_stats(dashboard_stats)}\n\n"
                f"{self.format_receipt_stats(receipt_stats)}\n\n"
                "Use /save to store this entry to your database."
            )
            await processing_message.edit_text(response, parse_mode=ParseMode.MARKDOWN)
            logger.info(f"Sent analysis results to user {user_id}")

            # Generate and send CSV file
            csv_file = self.create_csv_file(dashboard_stats, receipt_stats, note)
            await update.message.reply_document(
                document=csv_file,
                caption="Here is your data ready for Excel/Numbers",
                filename=csv_file.name,
            )

            # Send code block for quick view
            liters = (
                receipt_stats.total_cost / receipt_stats.price_per_liter
                if receipt_stats.price_per_liter and receipt_stats.total_cost
                else 0
            )

            csv_preview = (
                "```csv\n"
                "date;total-km;trip-km;liter;costs;euro_per_liter;consumption;note\n"
                f"{receipt_stats.date or ''};{dashboard_stats.total_km or ''};"
                f"{dashboard_stats.trip_km or ''};{liters:.2f};"
                f"{receipt_stats.total_cost or ''};{receipt_stats.price_per_liter or ''};"
                f"{dashboard_stats.consumption or ''};{note}\n"
                "```"
            )
            await update.message.reply_text(csv_preview, parse_mode=ParseMode.MARKDOWN)
            logger.info(f"Sent CSV data to user {user_id}")

        except Exception as e:
            logger.error(f"Error handling photo for user {user_id}: {str(e)}")
            if "processing_message" in locals():
                await processing_message.edit_text(
                    "An error occurred while processing your image."
                )
            else:
                await update.message.reply_text(
                    "An error occurred while processing your image."
                )

    async def handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle text messages."""
        logger.info(f"Received text message from user {update.effective_user.id}")
        await update.message.reply_text("Please send a photo instead of text.")

    def run(self):
        """Start the bot."""
        logger.info("Starting bot")

        # Create application with post_init for setting commands
        async def post_init(application):
            """Set bot commands for Telegram menu."""
            commands = [
                BotCommand("start", "Start the bot"),
                BotCommand("help", "Show available commands"),
                BotCommand("cancel", "Cancel current session"),
                BotCommand("save", "Save last scan to database"),
                BotCommand("history", "View recent fuel logs"),
                BotCommand("stats", "View fuel statistics"),
                BotCommand("export", "Export all data as CSV"),
                BotCommand("modify", "Edit a log entry"),
                BotCommand("delete", "Delete a log entry"),
                BotCommand("clear", "Clear all your data"),
            ]
            await application.bot.set_my_commands(commands)
            logger.info("Bot commands registered with Telegram")

        application = (
            Application.builder()
            .token(self.telegram_token)
            .post_init(post_init)
            .build()
        )

        # Modify conversation handler
        modify_handler = ConversationHandler(
            entry_points=[CommandHandler("modify", self.modify_start)],
            states={
                MODIFY_SELECT_FIELD: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, self.modify_select_field
                    )
                ],
                MODIFY_ENTER_VALUE: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, self.modify_enter_value
                    )
                ],
            },
            fallbacks=[
                CommandHandler("done", self.modify_done),
                CommandHandler("cancel", self.modify_cancel),
            ],
        )

        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("cancel", self.cancel))
        application.add_handler(CommandHandler("save", self.save))
        application.add_handler(CommandHandler("history", self.history))
        application.add_handler(CommandHandler("stats", self.stats))
        application.add_handler(CommandHandler("export", self.export))
        application.add_handler(CommandHandler("delete", self.delete))
        application.add_handler(CommandHandler("clear", self.clear))
        application.add_handler(modify_handler)
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text)
        )
        logger.info("Handlers added")

        # Start the bot
        logger.info("Bot is running")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    # Create and run the bot
    logger.info("Initializing bot")
    bot = TelegramBot()
    bot.run()
