import discord
from discord import app_commands
from dotenv import load_dotenv
import os
from processing import process_image
import ccimage
from pathlib import Path

load_dotenv()

MY_GUILD = discord.Object(id=os.getenv('GUILD'))
CC_COMP_DIR = os.getenv('CC_COMP_DIR')

class BotClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        self.tree.copy_global_to(guild=MY_GUILD)
        await self.tree.sync(guild=MY_GUILD)


intents = discord.Intents.default()
client = BotClient(intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('------')


@client.tree.command()
async def hello(interaction: discord.Interaction):
    """Says hello!"""
    await interaction.response.send_message(f'Hi, {interaction.user.mention}')

@client.tree.command(name = "displayimage", description = "My first application Command", guild=MY_GUILD) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def displayimage(interaction: discord.Interaction, attachment: discord.Attachment):
    print(attachment.content_type)
    typ = attachment.content_type.split(sep='/')
    if (typ[0] != "image"):
        await interaction.response.send_message("Bugger off, images only.")
        return

    await attachment.save(attachment.filename)

    await interaction.response.send_message("Processing...", file=discord.File(attachment.filename))
    image = ccimage.extend(attachment.filename)
    ccimage.init(image)
    data = ccimage.process(image, 50)

    p = Path(CC_COMP_DIR)
    for x in p.iterdir():
        if x.is_dir():       
            with open(str(x)+"/image", "w") as f:
                f.write(data)
    await interaction.edit_original_response(content="Done!")


client.run(os.getenv("BOT_TOKEN"))
