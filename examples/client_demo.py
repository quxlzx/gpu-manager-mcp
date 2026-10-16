#!/usr/bin/env python3
"""Minimal MCP client example: list GPUs and print a summary.

Run with the package installed:  python examples/client_demo.py
"""

import asyncio
import json

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    params = StdioServerParameters(command="gpu-manager-mcp", args=["stdio"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_system_gpu_summary", {})
            print(json.dumps(json.loads(result.content[0].text), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
