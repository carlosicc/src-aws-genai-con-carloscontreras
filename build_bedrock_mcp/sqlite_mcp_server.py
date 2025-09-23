#!/usr/bin/env python3
import asyncio, aiosqlite, os
from mcp.server.fastmcp import FastMCP

DB_PATH = os.path.join(os.path.dirname(__file__), "demo.db")
mcp = FastMCP("sqlite-demo-server")

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        """)
        await db.commit()

@mcp.tool()
async def list_users() -> list[dict]:
    """List all users from my app users"""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT id, name FROM users")
        rows = await cur.fetchall()
        await cur.close()
    return [{"id": r[0], "name": r[1]} for r in rows]

@mcp.tool()
async def add_user(name: str) -> str:
    """Add a new user in the app users table"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO users (name) VALUES (?)", (name,))
        await db.commit()
    return f"User {name} added."

if __name__ == "__main__":
    asyncio.run(init_db())
    asyncio.run(mcp.run_stdio_async())
