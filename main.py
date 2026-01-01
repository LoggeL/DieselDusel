import os
import logging
import json
import base64
from io import BytesIO
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
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

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        logger.info(f"Start command received from user {update.effective_user.id}")
        welcome_message = (
            "👋 Welcome to the Fuel Stats Bot!\n\n"
            "Send me a photo of your car's display, and I'll extract the following information:\n"
            "- Fuel consumption (l/100km)\n"
            "- Total kilometers\n"
            "- Trip kilometers\n\n"
            "Just send a photo to get started!"
        )
        await update.message.reply_text(welcome_message)
        logger.info(f"Welcome message sent to user {update.effective_user.id}")

    async def help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Send a message when the command /help is issued."""
        logger.info(f"Help command received from user {update.effective_user.id}")
        help_text = (
            "📸 Available Commands:\n\n"
            "/start - Start the bot and get welcome message\n"
            "/help - Show this help message\n"
            "/cancel - Cancel current image processing session\n\n"
            "How to use the bot:\n\n"
            "1. Take a clear photo of your car's display\n"
            "2. Send the photo to this bot\n"
            "3. Wait for the analysis results\n\n"
            "The photo should clearly show:\n"
            "- Fuel consumption (Verbrauch)\n"
            "- Total kilometers\n"
            "- Trip kilometers\n\n"
            "If you send the wrong image first, use /cancel to start over."
        )
        await update.message.reply_text(help_text)
        logger.info(f"Help message sent to user {update.effective_user.id}")

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Cancel the current image processing session."""
        user_id = update.effective_user.id
        logger.info(f"Cancel command received from user {user_id}")

        if "previous_stats" in context.user_data:
            context.user_data["previous_stats"] = None
            await update.message.reply_text(
                "✅ Session cancelled. You can now start over by sending a new image."
            )
        else:
            await update.message.reply_text(
                "ℹ️ No active session to cancel. You can start by sending an image."
            )
        logger.info(f"Session cancelled for user {user_id}")

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
                "model": "google/gemini-3-flash-preview",
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
            f"• ⛽ Consumption: {consumption} l/100km\n"
            f"• 🚗 Total Distance: {total_km} km\n"
            f"• 🛣 Trip Distance: {trip_km} km"
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

        return f"• 📅 Date: {date}\n• 💰 Price: {price} €/l\n• 💸 Total: {total} €"

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
                "🔄 Processing your image... Please wait."
            )
            logger.debug("Sent processing message")

            # If the message has text, add it as a note
            note = update.message.text if update.message.text else ""

            # Get the photo file
            photo = await update.message.photo[-1].get_file()
            image_bytes = await photo.download_as_bytearray()
            logger.debug("Photo downloaded")

            # Process the image
            result = self.process_image(image_bytes)
            if not result:
                logger.warning(f"Image processing failed for user {user_id}")
                await processing_message.edit_text(
                    "❌ Could not process the image. Please try again with a clearer photo."
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
                emoji = "🧾" if next_type == "receipt" else "🚗"

                await processing_message.edit_text(
                    f"✅ **{image_type.title()} Scanned**\n\n"
                    f"{formatted_stats}\n\n"
                    f"Please send the {emoji} {next_type} image next.",
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
                    "❌ Both images appear to be the same type. Please send one dashboard image and one receipt image."
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

            # Format the merged response message
            response = (
                "✅ **Analysis Complete!**\n\n"
                f"{self.format_dashboard_stats(dashboard_stats)}\n\n"
                f"{self.format_receipt_stats(receipt_stats)}"
            )
            await processing_message.edit_text(response, parse_mode=ParseMode.MARKDOWN)
            logger.info(f"Sent analysis results to user {user_id}")

            # Generate and send CSV file
            csv_file = self.create_csv_file(dashboard_stats, receipt_stats, note)
            await update.message.reply_document(
                document=csv_file,
                caption="📊 Here is your data ready for Excel/Numbers",
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
                    "❌ An error occurred while processing your image."
                )
            else:
                await update.message.reply_text(
                    "❌ An error occurred while processing your image."
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
        # Create application
        application = Application.builder().token(self.telegram_token).build()

        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("cancel", self.cancel))
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
