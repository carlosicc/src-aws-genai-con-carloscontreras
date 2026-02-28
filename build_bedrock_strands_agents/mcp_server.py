from mcp.server.fastmcp import FastMCP
from ddgs import DDGS
from ddgs.exceptions import RatelimitException, DDGSException

mcp = FastMCP(host="0.0.0.0", stateless_http=True)

# Define a websearch tool
@mcp.tool()
def websearch(keywords: str, region: str = "us-en", max_results: int | None = None) -> list:
    """Search the web to get updated information.
    Args:
        keywords (str): The search query keywords.
        region (str): The search region: wt-wt, us-en, uk-en, ru-ru, etc..
        max_results (int | None): The maximum number of results to return.
    Returns:
        List of dictionaries with search results.
    """
    try:
        results = DDGS().text(keywords, region=region, max_results=max_results)
        return results if results else "No results found."
    except RatelimitException:
        return "RatelimitException: Please try again after a short delay."
    except DDGSException as d:
        return f"DuckDuckGoSearchException: {d}"
    except Exception as e:
        return f"Exception: {e}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
