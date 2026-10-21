from redbot.core.bot import Red

from .ballping import BallPing


async def setup(bot: Red) -> None:
    await bot.add_cog(BallPing(bot))
