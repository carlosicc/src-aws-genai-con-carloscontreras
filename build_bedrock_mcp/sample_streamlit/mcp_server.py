from mcp.server.fastmcp import FastMCP

mcp = FastMCP(host="0.0.0.0", stateless_http=True)

@mcp.tool()
async def executeTrade(ticker, quantity, price):
    """
    Execute a trade for the given ticker, quantity, and price.
    
    Sample input:
    {
        "ticker": "AMZN",
        "quantity": 1000,
        "price": 150.25
    }

    """
    # Simulate trade execution
    return {
        "tradeId": "T12345",
        "status": "Executed",
        "timestamp": "2025-04-09T22:58:00"
    }
    
@mcp.tool()
async def sendTradeDetails(tradeId):
    """
    Send trade details for the given tradeId.

    Sample input:
    {
        "tradeId": "T12345"
    }
    """
    return {
        "status": "Details Sent",
        "recipientSystem": "MiddleOffice",
        "timestamp": "2025-04-09T22:59:00"
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")