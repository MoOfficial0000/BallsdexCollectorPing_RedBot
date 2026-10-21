import asyncio
import difflib
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from urllib.parse import urlparse

import aiohttp
import discord
import numpy as np
from PIL import Image, ImageOps
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.data_manager import bundled_data_path, cog_data_path
from redbot.core.utils.chat_formatting import escape, pagify

log = logging.getLogger("red.ballping")

HASH_BITS = 20
HASH_SIZE = 100
HASH_LEN = HASH_BITS * HASH_BITS // 4  # 100 hex characters
MAX_BYTES = 10 * 1024 * 1024
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
NO_MENTIONS = discord.AllowedMentions.none()


class BadImage(Exception):
    """Raised when bytes can't be decoded as an image."""


class Index(NamedTuple):
    names: List[str]  # one entry per stored hash (a ball can have several)
    matrix: np.ndarray  # shape (len(names), HASH_LEN), uint8 hex characters
    lookup: Dict[str, str]  # lowercase name -> canonical name


class FetchError(Exception):
    """Raised when an image can't be downloaded."""


class Source(NamedTuple):
    kind: str  # "attachment", "embed image", "embed thumbnail" or "component"
    url: str
    filename: str
    attachment: Optional[discord.Attachment] = None


DISCORD_HOSTS = ("discordapp.com", "discordapp.net", "discord.com")


def _is_discord_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme in ("http", "https") and any(
        host == d or host.endswith("." + d) for d in DISCORD_HOSTS
    )


def _name_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path) or "image.png"


def _is_interaction_response(message: discord.Message) -> bool:
    """True for replies to slash commands/buttons (e.g. a collection embed), never for spawns."""
    if hasattr(message, "interaction_metadata"):
        return message.interaction_metadata is not None
    return getattr(message, "interaction", None) is not None


def _walk(node, media: List[Tuple[str, str]], names: List[str], depth: int = 0) -> None:
    """Collect media URLs and component type names from a message's components.

    Duck-typed on purpose: Discord's newer layout components differ between discord.py versions.
    """
    if node is None or depth > 6:
        return
    if isinstance(node, (list, tuple)):
        for child in node:
            _walk(child, media, names, depth)
        return
    names.append(type(node).__name__)
    item = getattr(node, "media", None)
    url = getattr(item, "proxy_url", None) or getattr(item, "url", None)
    if url:
        media.append((url, _name_from_url(url)))
    for attr in ("children", "items", "components"):
        child = getattr(node, attr, None)
        if isinstance(child, (list, tuple)):
            _walk(child, media, names, depth + 1)
    _walk(getattr(node, "accessory", None), media, names, depth + 1)


def image_hash(data: bytes) -> str:
    """Python port of BallIdentifier's hash: sharp resize(100,100) -> black flatten -> image-hash(20 bits)."""
    with Image.open(io.BytesIO(data)) as src:
        rgba = src.convert("RGBA")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    fitted = ImageOps.fit(rgba, (HASH_SIZE, HASH_SIZE), method=resample, centering=(0.5, 0.5))
    canvas = Image.new("RGBA", fitted.size, (0, 0, 0, 255))
    canvas.alpha_composite(fitted)

    rgb = np.asarray(canvas.convert("RGB"), dtype=np.int64)
    block = HASH_SIZE // HASH_BITS
    sums = (
        rgb.sum(axis=2)
        .reshape(HASH_BITS, block, HASH_BITS, block)
        .sum(axis=(1, 3))
        .reshape(-1)
        .astype(np.float64)
    )
    half = block * block * 256 * 3 / 2
    band = sums.size // 4
    bits = np.zeros(sums.size, dtype=np.int64)
    for i in range(4):
        seg = sums[i * band : (i + 1) * band]
        ordered = np.sort(seg)
        # image-hash's median() reads [n/2] and [n/2 + 1] for even n; kept so hashes stay comparable.
        median = (ordered[band // 2] + ordered[band // 2 + 1]) / 2.0
        bits[i * band : (i + 1) * band] = (
            (seg > median) | ((np.abs(seg - median) < 1) & (median > half))
        ).astype(np.int64)
    nibbles = bits.reshape(-1, 4) @ np.array([8, 4, 2, 1])
    return "".join(format(int(n), "x") for n in nibbles)


def _load_json(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class BallPing(commands.Cog):
    """Identify Ballsdex spawns from the spawn art and ping the people who want them."""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(
            self, identifier=482916037524180913, force_registration=True
        )
        self.config.register_guild(
            enabled=True,
            spawn_bots=[],  # bot user IDs whose image messages count as spawns
            channels=[],  # empty = listen in every channel
            max_diff=30,  # max hex-character mismatches (of 100) to accept a match
            announce_all=False,  # also post when nobody is subscribed / unknown
            pings={},  # {ball name: {"users": [ids], "roles": [ids]}}
        )
        # Art uploaded by the owner; global because spawn art is the same in every server.
        self.config.register_global(art_overrides={})  # {ball: [{hash, file, by, at}]}

        self._base: Dict[str, str] = {}
        self._index: Optional[Index] = None
        self._index_lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None

    async def cog_load(self) -> None:
        self._session = aiohttp.ClientSession()
        path = bundled_data_path(self) / "BallsdexHashes.json"
        self._base = await self.bot.loop.run_in_executor(None, _load_json, path)
        await self._rebuild_index()

    async def cog_unload(self) -> None:
        if self._session is not None:
            await self._session.close()

    # ------------------------------------------------------------------ index

    async def _rebuild_index(self) -> None:
        # Read config *inside* the lock so a slower rebuild can never overwrite a newer one.
        async with self._index_lock:
            overrides = await self.config.art_overrides()
            rows = list(self._base.items())
            for name, entries in overrides.items():
                rows.extend((name, e["hash"]) for e in entries)
            rows = [(n, h) for n, h in rows if len(h) == HASH_LEN]
            names = [n for n, _ in rows]
            joined = "".join(h for _, h in rows).encode("ascii")
            matrix = np.frombuffer(joined, dtype=np.uint8).reshape(len(rows), HASH_LEN)
            lookup = {n.lower(): n for n in names}
            self._index = Index(names, matrix, lookup)  # single atomic swap

    def _match(self, h: str, k: int = 3) -> List[Tuple[str, int]]:
        index = self._index
        if index is None or not index.names:
            return []
        query = np.frombuffer(h.encode("ascii"), dtype=np.uint8)
        diffs = (index.matrix != query).sum(axis=1)
        out: List[Tuple[str, int]] = []
        seen = set()
        for i in np.argsort(diffs, kind="stable"):
            name = index.names[int(i)]
            if name in seen:
                continue
            seen.add(name)
            out.append((name, int(diffs[i])))
            if len(out) == k:
                break
        return out

    async def _hash(self, data: bytes) -> str:
        try:
            return await self.bot.loop.run_in_executor(None, image_hash, data)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
            raise BadImage(str(e)) from e

    def _resolve(self, name: str) -> Optional[str]:
        index = self._index
        return index.lookup.get(name.strip().lower()) if index else None

    async def _need_ball(self, ctx: commands.Context, name: str) -> Optional[str]:
        """Resolve a ball name or tell the user it's unknown (with suggestions)."""
        found = self._resolve(name)
        if found:
            return found
        index = self._index
        close = difflib.get_close_matches(name.strip().lower(), index.lookup.keys(), n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(index.lookup[c] for c in close)}?" if close else ""
        await ctx.send(escape(f"I don't know a ball called '{name}'.{hint}", mass_mentions=True))
        return None

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _is_image(att: discord.Attachment) -> bool:
        ctype = att.content_type or ""
        return (
            ctype.startswith("image/")
            or att.filename.lower().endswith(IMAGE_EXTS)
            or ctype in ("", "application/octet-stream")
        )

    def _sources(self, message: discord.Message, *, embeds: bool = True) -> List[Source]:
        """Images we can read from a message: attachments first, else embeds (optional), else components.

        Automatic spawn detection passes embeds=False: embeds are info cards (e.g. a collection
        thumbnail), not spawns.
        """
        found = [
            Source("attachment", a.url, a.filename, a)
            for a in message.attachments
            if self._is_image(a)
        ]
        if found:
            return found
        for emb in message.embeds if embeds else []:
            for kind, part in (("embed image", emb.image), ("embed thumbnail", emb.thumbnail)):
                url = getattr(part, "proxy_url", None) or getattr(part, "url", None)
                if url:
                    found.append(Source(kind, url, _name_from_url(url)))
        media: List[Tuple[str, str]] = []
        _walk(message.components, media, [])
        found.extend(Source("component", url, name) for url, name in media)
        return found

    async def _raw_component_sources(
        self, message: discord.Message
    ) -> Tuple[List[Source], List[str]]:
        """Read raw Components V2 payloads when this discord.py version drops them.

        Older discord.py releases may parse a Components V2 message as having no
        attachments/embeds and discard its media components. Fetching the message
        payload directly preserves the component JSON, including media-gallery
        and container images.
        """
        try:
            payload = await self.bot.http.get_message(message.channel.id, message.id)
        except (discord.HTTPException, AttributeError) as e:
            log.debug("Could not fetch raw payload for message %s: %s", message.id, e)
            return [], []

        found: List[Source] = []
        names: List[str] = []
        seen = set()

        def walk_raw(node, depth: int = 0) -> None:
            if node is None or depth > 12:
                return
            if isinstance(node, list):
                for child in node:
                    walk_raw(child, depth + 1)
                return
            if not isinstance(node, dict):
                return

            component_type = node.get("type")
            if component_type is not None:
                names.append(f"RawType{component_type}")

            # Components V2 image/media-gallery items expose their CDN URL here.
            media = node.get("media")
            if isinstance(media, dict):
                url = media.get("proxy_url") or media.get("url")
                if isinstance(url, str) and _is_discord_url(url) and url not in seen:
                    seen.add(url)
                    found.append(Source("component", url, _name_from_url(url)))

            # Some payload variants expose the media URL directly on the image node.
            if component_type in (11, 12):
                url = node.get("proxy_url") or node.get("url")
                if isinstance(url, str) and _is_discord_url(url) and url not in seen:
                    seen.add(url)
                    found.append(Source("component", url, _name_from_url(url)))

            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk_raw(value, depth + 1)

        walk_raw(payload.get("components", []))
        return found, names

    async def _sources_with_raw(
        self, message: discord.Message, *, embeds: bool = True
    ) -> Tuple[List[Source], List[str]]:
        """Get parsed sources, falling back to raw V2 components if needed."""
        sources = self._sources(message, embeds=embeds)
        names: List[str] = []
        _walk(message.components, [], names)
        # Always inspect raw components if the parsed message has no component
        # media; attachments/embeds may exist alongside a V2 spawn image.
        if not any(src.kind == "component" for src in sources):
            raw_sources, raw_names = await self._raw_component_sources(message)
            existing_urls = {src.url for src in sources}
            sources.extend(src for src in raw_sources if src.url not in existing_urls)
            names.extend(n for n in raw_names if n not in names)
        return sources, names

    async def _fetch(self, src: Source) -> bytes:
        try:
            if src.attachment is not None:
                if src.attachment.size > MAX_BYTES:
                    raise FetchError("image too large")
                return await src.attachment.read()
            # Only Discord's own CDN/proxy hosts, so a crafted embed can't make the bot fetch arbitrary URLs.
            if self._session is None or not _is_discord_url(src.url):
                raise FetchError("image isn't hosted on Discord")
            timeout = aiohttp.ClientTimeout(total=15)
            async with self._session.get(src.url, timeout=timeout) as resp:
                if resp.status != 200:
                    raise FetchError(f"HTTP {resp.status}")
                buf = bytearray()
                async for chunk in resp.content.iter_chunked(65536):
                    buf += chunk
                    if len(buf) > MAX_BYTES:
                        raise FetchError("image too large")
                return bytes(buf)
        except (discord.HTTPException, aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise FetchError(str(e) or type(e).__name__) from e

    async def _referenced(self, ctx: commands.Context) -> Optional[discord.Message]:
        """The message the command message replies to, if it can be read."""
        ref = ctx.message.reference
        if ref is None:
            return None
        if isinstance(ref.resolved, discord.Message):
            return ref.resolved
        if ref.message_id is None:
            return None
        try:
            return await ctx.channel.fetch_message(ref.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return None

    async def _get_image(self, ctx: commands.Context) -> Optional[Tuple[bytes, str]]:
        """Image from the command message, or from the message it replies to."""
        sources, _ = await self._sources_with_raw(ctx.message)
        replied = False
        if not sources and ctx.message.reference:
            replied = True
            ref = await self._referenced(ctx)
            if ref is None:
                await ctx.send(
                    "I couldn't read the message you replied to (I need Read Message History there)."
                )
                return None
            sources, _ = await self._sources_with_raw(ref)
        if not sources:
            if replied:
                await ctx.send(
                    f"That message has no image I can read. Run `{ctx.clean_prefix}ballping inspect` "
                    "as a reply to it to see what it contains."
                )
            else:
                await ctx.send("Attach an image, or reply to a message that has one.")
            return None
        try:
            return await self._fetch(sources[0]), sources[0].filename
        except FetchError as e:
            await ctx.send(escape(f"I couldn't download that image ({e}).", mass_mentions=True))
            return None

    async def _identify_message(self, message: discord.Message) -> Optional[Tuple[str, int]]:
        """Best (name, diff) across up to three images in a message."""
        best: Optional[Tuple[str, int]] = None
        sources, _ = await self._sources_with_raw(message, embeds=False)
        for src in sources[:3]:
            try:
                results = self._match(await self._hash(await self._fetch(src)))
            except (FetchError, BadImage) as e:
                log.debug("Skipping %s in message %s: %s", src.kind, message.id, e)
                continue
            if results and (best is None or results[0][1] < best[1]):
                best = results[0]
        return best

    @staticmethod
    def _mentions(entry: Optional[dict]) -> str:
        if not entry:
            return ""
        return " ".join(
            [f"<@{u}>" for u in entry.get("users", [])]
            + [f"<@&{r}>" for r in entry.get("roles", [])]
        )

    @staticmethod
    async def _say(ctx: commands.Context, text: str) -> None:
        await ctx.send(text, allowed_mentions=NO_MENTIONS)

    # --------------------------------------------------------------- listener

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or not message.author.bot or self._index is None:
            return
        if message.author.id == self.bot.user.id or _is_interaction_response(message):
            return
        if await self.bot.cog_disabled_in_guild(self, message.guild):
            return
        conf = await self.config.guild(message.guild).all()
        if not conf["enabled"] or message.author.id not in conf["spawn_bots"]:
            return
        if conf["channels"] and message.channel.id not in conf["channels"]:
            return
        best = await self._identify_message(message)
        if best is None:
            return

        name, diff = best
        match_pct = 100 - diff
        if diff > conf["max_diff"]:
            if not conf["announce_all"]:
                return
            text = f"Couldn't identify this spawn (closest: **{name}**, {match_pct}% match)."
        else:
            mentions = self._mentions(conf["pings"].get(name))
            if not mentions and not conf["announce_all"]:
                return
            text = f"**{name}** spawned!"
            if mentions:
                text += f" {mentions}"
            if conf["announce_all"]:
                text += f" ({match_pct}% match)"

        allowed = discord.AllowedMentions(
            everyone=False, users=True, roles=True, replied_user=False
        )
        try:
            await message.channel.send(
                text,
                reference=message.to_reference(fail_if_not_exists=False),
                allowed_mentions=allowed,
            )
        except discord.Forbidden:
            log.debug("Missing permission to send in channel %s", message.channel.id)
        except discord.HTTPException as e:
            log.warning("Failed to send spawn ping in channel %s: %s", message.channel.id, e)

    # --------------------------------------------------------------- commands

    @commands.group(name="ballping", aliases=["bp"])
    @commands.guild_only()
    @commands.admin_or_permissions(manage_guild=True)
    async def ballping(self, ctx: commands.Context):
        """Configure spawn identification and pings for this server."""

    @ballping.command(name="status")
    async def ballping_status(self, ctx: commands.Context):
        """Show this server's current BallPing settings."""
        conf = await self.config.guild(ctx.guild).all()
        bots = " ".join(f"<@{b}>" for b in conf["spawn_bots"]) or "none set"
        channels = " ".join(f"<#{c}>" for c in conf["channels"]) or "all channels"
        text = (
            f"Enabled: {conf['enabled']}\n"
            f"Spawn bots: {bots}\n"
            f"Channels: {channels}\n"
            f"Max diff: {conf['max_diff']} (min {100 - conf['max_diff']}% match)\n"
            f"Announce every spawn: {conf['announce_all']}\n"
            f"Balls with pings: {len(conf['pings'])}\n"
            f"Known balls: {len(self._index.lookup) if self._index else 0}"
        )
        await self._say(ctx, text)

    @ballping.command(name="toggle")
    async def ballping_toggle(self, ctx: commands.Context):
        """Turn spawn identification on or off for this server."""
        async with self.config.guild(ctx.guild).enabled.get_lock():
            new = not await self.config.guild(ctx.guild).enabled()
            await self.config.guild(ctx.guild).enabled.set(new)
        await ctx.send(f"BallPing is now {'enabled' if new else 'disabled'} in this server.")

    @ballping.command(name="announce")
    async def ballping_announce(self, ctx: commands.Context):
        """Toggle posting an identification for every spawn, even with nobody subscribed."""
        async with self.config.guild(ctx.guild).announce_all.get_lock():
            new = not await self.config.guild(ctx.guild).announce_all()
            await self.config.guild(ctx.guild).announce_all.set(new)
        await ctx.send(
            f"I will {'now' if new else 'no longer'} announce every spawn (useful for testing)."
        )

    @ballping.command(name="maxdiff")
    async def ballping_maxdiff(self, ctx: commands.Context, value: int):
        """Set the match tolerance (0-60). Lower is stricter; 30 means at least a 70% match."""
        if not 0 <= value <= 60:
            await ctx.send("Pick a value between 0 and 60.")
            return
        await self.config.guild(ctx.guild).max_diff.set(value)
        await ctx.send(f"Max diff set to {value} (matches below {100 - value}% are ignored).")

    @ballping.group(name="spawnbot")
    async def ballping_spawnbot(self, ctx: commands.Context):
        """Manage which bots' image messages are treated as spawns."""

    @ballping_spawnbot.command(name="add")
    async def spawnbot_add(self, ctx: commands.Context, bot: discord.User):
        """Treat image messages from this bot as spawns."""
        if not bot.bot:
            await ctx.send("That account isn't a bot.")
            return
        async with self.config.guild(ctx.guild).spawn_bots() as bots:
            if bot.id in bots:
                await ctx.send(f"{escape(str(bot), mass_mentions=True)} is already listed.")
                return
            bots.append(bot.id)
        await ctx.send(f"Now watching {escape(str(bot), mass_mentions=True)} for spawns.")

    @ballping_spawnbot.command(name="remove")
    async def spawnbot_remove(self, ctx: commands.Context, bot: discord.User):
        """Stop treating this bot's messages as spawns."""
        async with self.config.guild(ctx.guild).spawn_bots() as bots:
            if bot.id not in bots:
                await ctx.send("That bot isn't in the list.")
                return
            bots.remove(bot.id)
        await ctx.send(f"No longer watching {escape(str(bot), mass_mentions=True)}.")

    @ballping_spawnbot.command(name="list")
    async def spawnbot_list(self, ctx: commands.Context):
        """List the watched spawn bots."""
        bots = await self.config.guild(ctx.guild).spawn_bots()
        await self._say(ctx, " ".join(f"<@{b}>" for b in bots) or "No spawn bots set.")

    @ballping.group(name="channel")
    async def ballping_channel(self, ctx: commands.Context):
        """Limit spawn detection to specific channels (none set = all channels)."""

    @ballping_channel.command(name="add")
    async def channel_add(self, ctx: commands.Context, channel: discord.TextChannel):
        """Only detect spawns in this channel (can be repeated for more)."""
        async with self.config.guild(ctx.guild).channels() as chans:
            if channel.id in chans:
                await ctx.send(f"{channel.mention} is already listed.")
                return
            chans.append(channel.id)
        await ctx.send(f"Now detecting spawns in {channel.mention}.")

    @ballping_channel.command(name="remove")
    async def channel_remove(self, ctx: commands.Context, channel: discord.TextChannel):
        """Remove a channel from the list."""
        async with self.config.guild(ctx.guild).channels() as chans:
            if channel.id not in chans:
                await ctx.send("That channel isn't in the list.")
                return
            chans.remove(channel.id)
        await ctx.send(f"Removed {channel.mention}. (If the list is now empty, all channels count.)")

    @ballping.group(name="ping")
    async def ballping_ping(self, ctx: commands.Context):
        """Choose who gets pinged for which ball."""

    @ballping_ping.command(name="add")
    async def ping_add(
        self,
        ctx: commands.Context,
        target: Union[discord.Member, discord.Role],
        *,
        ball: str,
    ):
        """Ping a member or role when a ball spawns.

        Example: `[p]ballping ping add @Sam Spanish Empire`
        """
        if isinstance(target, discord.Role) and target.is_default():
            await ctx.send("I won't ping @everyone.")
            return
        name = await self._need_ball(ctx, ball)
        if name is None:
            return
        key = "roles" if isinstance(target, discord.Role) else "users"
        async with self.config.guild(ctx.guild).pings() as pings:
            entry = pings.setdefault(name, {"users": [], "roles": []})
            if target.id in entry[key]:
                await ctx.send(f"They're already subscribed to **{name}**.")
                return
            entry[key].append(target.id)
        label = escape(target.name if isinstance(target, discord.Role) else target.display_name, mass_mentions=True)
        await self._say(ctx, f"{label} will be pinged when **{name}** spawns.")

    @ballping_ping.command(name="remove")
    async def ping_remove(
        self,
        ctx: commands.Context,
        target: Union[discord.Member, discord.Role],
        *,
        ball: str,
    ):
        """Stop pinging a member or role for a ball."""
        name = await self._need_ball(ctx, ball)
        if name is None:
            return
        key = "roles" if isinstance(target, discord.Role) else "users"
        async with self.config.guild(ctx.guild).pings() as pings:
            entry = pings.get(name)
            if not entry or target.id not in entry[key]:
                await ctx.send(f"They aren't subscribed to **{name}**.")
                return
            entry[key].remove(target.id)
            if not entry["users"] and not entry["roles"]:
                del pings[name]
        label = escape(target.name if isinstance(target, discord.Role) else target.display_name, mass_mentions=True)
        await self._say(ctx, f"{label} will no longer be pinged for **{name}**.")

    @ballping_ping.command(name="clear")
    async def ping_clear(self, ctx: commands.Context, *, ball: str):
        """Remove every subscription for a ball."""
        name = await self._need_ball(ctx, ball)
        if name is None:
            return
        async with self.config.guild(ctx.guild).pings() as pings:
            if pings.pop(name, None) is None:
                await ctx.send(f"Nobody was subscribed to **{name}**.")
                return
        await ctx.send(f"Cleared all subscriptions for **{name}**.")

    @ballping_ping.command(name="list")
    async def ping_list(self, ctx: commands.Context, *, ball: Optional[str] = None):
        """List subscriptions for one ball, or for every ball in this server."""
        pings = await self.config.guild(ctx.guild).pings()
        if ball:
            name = await self._need_ball(ctx, ball)
            if name is None:
                return
            pings = {name: pings[name]} if name in pings else {}
        if not pings:
            await ctx.send("No subscriptions found.")
            return
        lines = [f"**{n}**: {self._mentions(e)}" for n, e in sorted(pings.items())]
        for page in pagify("\n".join(lines)):
            await self._say(ctx, page)

    @ballping.command(name="test")
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def ballping_test(self, ctx: commands.Context):
        """Identify an image without pinging anyone (attach one or reply to one)."""
        got = await self._get_image(ctx)
        if got is None:
            return
        async with ctx.typing():
            try:
                results = self._match(await self._hash(got[0]))
            except BadImage:
                await ctx.send("I couldn't read that as an image.")
                return
        max_diff = await self.config.guild(ctx.guild).max_diff()
        lines = [f"{i}. **{n}**: {100 - d}% match (diff {d})" for i, (n, d) in enumerate(results, 1)]
        verdict = "Would be accepted." if results and results[0][1] <= max_diff else "Below the match threshold."
        await ctx.send(escape("\n".join(lines) + f"\n{verdict}", mass_mentions=True))

    @ballping.command(name="inspect")
    async def ballping_inspect(self, ctx: commands.Context):
        """Reply to a spawn message with this to see what image data I can find in it."""
        if not ctx.message.reference:
            await ctx.send("Reply to the spawn message with this command.")
            return
        msg = await self._referenced(ctx)
        if msg is None:
            await ctx.send("I couldn't read that message (I need Read Message History there).")
            return
        conf = await self.config.guild(ctx.guild).all()
        media: List[Tuple[str, str]] = []
        names: List[str] = []
        _walk(msg.components, media, names)
        sources, raw_names = await self._sources_with_raw(msg)
        names.extend(n for n in raw_names if n not in names)
        spawn_sources, _ = await self._sources_with_raw(msg, embeds=False)
        embeds = [
            f"image={'yes' if getattr(e.image, 'url', None) else 'no'}, "
            f"thumbnail={'yes' if getattr(e.thumbnail, 'url', None) else 'no'}"
            for e in msg.embeds
        ]
        listed = msg.author.id in conf["spawn_bots"]
        chan_ok = not conf["channels"] or msg.channel.id in conf["channels"]
        slash = _is_interaction_response(msg)
        reasons = []
        if not msg.author.bot:
            reasons.append("author isn't a bot")
        if not listed:
            reasons.append(f"author isn't a spawn bot here (use `{ctx.clean_prefix}ballping spawnbot add`)")
        if not conf["enabled"]:
            reasons.append("BallPing is disabled in this server")
        if not chan_ok:
            reasons.append("channel isn't in the allowed list")
        if slash:
            reasons.append("it's a slash-command/button response, which is ignored")
        if not spawn_sources:
            reasons.append("no attached or layout image (embed images are ignored)")
        lines = [
            f"Author: {msg.author} ({msg.author.id}), bot: {msg.author.bot}",
            f"Listed as a spawn bot: {listed}",
            f"Channel allowed: {chan_ok}",
            f"Slash-command/button response: {slash}",
            f"Attachments: {len(msg.attachments)} "
            + ", ".join(f"{a.filename} [{a.content_type}]" for a in msg.attachments),
            f"Embeds: {len(msg.embeds)} " + "; ".join(embeds),
            f"Components: {', '.join(names) if names else 'none'}",
            f"Images readable (for {ctx.clean_prefix}ballping test/art): {len(sources)}",
            f"Would be detected as a spawn: {not reasons}",
        ]
        if reasons:
            lines.append("Why not: " + "; ".join(reasons) + ".")
        if not sources and not msg.attachments and not msg.embeds and not names:
            lines.append(
                "I see nothing at all on that message. If you can see an image on it, the bot's "
                "Message Content intent may be off."
            )
        await ctx.send(escape("\n".join(lines), mass_mentions=True))

    # ---------------------------------------------------------- art (global)

    @ballping.group(name="art")
    @commands.is_owner()
    async def ballping_art(self, ctx: commands.Context):
        """Bot owner: teach the identifier new or updated spawn art (applies to all servers)."""

    async def _store_art(self, ctx: commands.Context, name: str) -> None:
        got = await self._get_image(ctx)
        if got is None:
            return
        data, filename = got
        async with ctx.typing():
            try:
                h = await self._hash(data)
            except BadImage:
                await ctx.send("I couldn't read that as an image.")
                return
            before = self._match(h, 1)
            if before and before[0][0] == name and before[0][1] <= 2:
                await ctx.send(f"That art is already recognised as **{name}**; nothing to add.")
                return
            ext = Path(filename).suffix.lower()
            ext = ext if ext in IMAGE_EXTS else ".png"
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name)[:40]
            file_name = f"{safe}_{int(time.time())}{ext}"
            try:
                await self.bot.loop.run_in_executor(
                    None, _write_bytes, cog_data_path(self) / "art" / file_name, data
                )
            except OSError as e:
                log.warning("Could not save art image: %s", e)
                file_name = ""
        entry = {"hash": h, "file": file_name, "by": ctx.author.id, "at": int(time.time())}
        async with self.config.art_overrides() as overrides:
            overrides.setdefault(name, []).append(entry)
            count = len(overrides[name])
        await self._rebuild_index()
        note = ""
        if before and before[0][0] != name and before[0][1] <= 10:
            note = f"\nHeads up: before adding, this looked like **{before[0][0]}** ({100 - before[0][1]}% match). Check the name."
        await ctx.send(
            escape(f"Added new spawn art for **{name}** ({count} extra variant(s) stored).{note}", mass_mentions=True)
        )

    @ballping_art.command(name="add")
    async def art_add(self, ctx: commands.Context, *, ball: str):
        """Add new spawn art for an existing ball (attach an image or reply to one)."""
        name = await self._need_ball(ctx, ball)
        if name is None:
            await ctx.send("If it's a brand-new ball, use `art new <name>` instead.")
            return
        await self._store_art(ctx, name)

    @ballping_art.command(name="new")
    async def art_new(self, ctx: commands.Context, *, ball: str):
        """Register a ball that isn't in the built-in list, using its spawn art."""
        name = " ".join(ball.split())[:60]
        if not name:
            await ctx.send("Give the ball a name.")
            return
        if self._resolve(name):
            await ctx.send(f"**{self._resolve(name)}** already exists; use `art add` instead.")
            return
        await self._store_art(ctx, name)

    @ballping_art.command(name="list")
    async def art_list(self, ctx: commands.Context, *, ball: Optional[str] = None):
        """List uploaded art, or the variants stored for one ball."""
        overrides = await self.config.art_overrides()
        if ball:
            name = await self._need_ball(ctx, ball)
            if name is None:
                return
            entries = overrides.get(name, [])
            if not entries:
                await ctx.send(f"No uploaded art for **{name}** (only the built-in hash).")
                return
            lines = [f"{i}. added <t:{e['at']}:d> by <@{e['by']}>" for i, e in enumerate(entries, 1)]
            await self._say(ctx, f"**{name}**\n" + "\n".join(lines))
            return
        if not overrides:
            await ctx.send("No art has been uploaded yet.")
            return
        lines = [f"**{n}**: {len(e)} variant(s)" for n, e in sorted(overrides.items())]
        for page in pagify("\n".join(lines)):
            await self._say(ctx, page)

    @ballping_art.command(name="remove")
    async def art_remove(self, ctx: commands.Context, number: int, *, ball: str):
        """Remove an uploaded art variant by its number from `art list <ball>`."""
        name = await self._need_ball(ctx, ball)
        if name is None:
            return
        async with self.config.art_overrides() as overrides:
            entries = overrides.get(name, [])
            if not 1 <= number <= len(entries):
                await ctx.send(f"**{name}** has {len(entries)} uploaded variant(s).")
                return
            removed = entries.pop(number - 1)
            if not entries:
                overrides.pop(name, None)
        await self._rebuild_index()
        if removed.get("file"):
            try:
                (cog_data_path(self) / "art" / removed["file"]).unlink()
            except OSError:
                pass
        await ctx.send(f"Removed variant {number} for **{name}**.")

    # --------------------------------------------------------- data deletion

    async def red_delete_data_for_user(self, *, requester: str, user_id: int) -> None:
        for gid, data in (await self.config.all_guilds()).items():
            pings = data.get("pings", {})
            changed = False
            for name in list(pings):
                if user_id in pings[name]["users"]:
                    pings[name]["users"].remove(user_id)
                    changed = True
                if not pings[name]["users"] and not pings[name]["roles"]:
                    del pings[name]
            if changed:
                await self.config.guild_from_id(gid).pings.set(pings)
        async with self.config.art_overrides() as overrides:
            for entries in overrides.values():
                for e in entries:
                    if e.get("by") == user_id:
                        e["by"] = 0
