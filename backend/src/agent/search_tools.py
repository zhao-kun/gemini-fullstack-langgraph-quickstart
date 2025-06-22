"""Search tools for web research functionality."""

import re
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Represents a search result from any search provider."""
    title: str
    url: str
    content: str
    snippet: str
    source: str = "unknown"


class BraveSearchTool:
    """Search tool using Brave Search API for web search and content retrieval."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.search.brave.com"):
        """Initialize Brave search tool.
        
        Args:
            api_key: Brave Search API key
            base_url: Brave Search API base URL
        """
        if not api_key:
            raise ValueError("Brave Search API key cannot be empty")
            
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': api_key
        })
    
    def search_and_scrape(self, query: str, max_results: int = 5, max_content_length: int = 4000) -> List[SearchResult]:
        """Search using Brave Search API and extract content.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            max_content_length: Maximum content length per result
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Get search results from Brave Search API
            search_results = self._brave_search(query, max_results)
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                # Extract basic content from snippet and description
                content = self._extract_content_from_result(result, max_content_length)
                
                results.append(SearchResult(
                    title=result.get('title', 'No Title'),
                    url=result.get('url', ''),
                    content=content,
                    snippet=result.get('description', '')[:200] + '...' if result.get('description') else '',
                    source='brave'
                ))
            
            return results[:max_results]
            
        except Exception as e:
            print(f"Brave search error: {str(e)}")
            return []
    
    def _brave_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Brave Search API."""
        try:
            response = self.session.get(
                f"{self.base_url}/res/v1/web/search",
                params={
                    'q': query,
                    'count': min(max_results, 20),  # Brave API supports up to 20 results
                    'country': 'US',
                    'search_lang': 'en',
                    'ui_lang': 'en-US',
                    'text_decorations': False,
                    'spellcheck': True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                web_results = data.get('web', {}).get('results', [])
                return web_results
            else:
                raise Exception(f"Brave Search API returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Brave Search API error: {str(e)}")
            return []
    
    def _extract_content_from_result(self, result: Dict[str, Any], max_content_length: int) -> str:
        """Extract and format content from Brave search result."""
        content_parts = []
        
        # Add description if available
        if result.get('description'):
            content_parts.append(result['description'])
        
        # Add any extra snippets if available
        if result.get('extra_snippets'):
            for snippet in result['extra_snippets']:
                content_parts.append(snippet)
        
        # Add meta description if available
        if result.get('meta_url', {}).get('description'):
            content_parts.append(result['meta_url']['description'])
        
        # Combine content
        content = ' '.join(content_parts)
        
        # Clean and truncate content
        content = self._clean_content(content)
        if len(content) > max_content_length:
            content = content[:max_content_length] + '...'
        
        return content
    
    def format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = f"""
**Source {i}: {result.title}**
URL: {result.url}
Content: {result.content}

---
"""
            formatted_results.append(formatted_result)
        
        return "\n".join(formatted_results)
    
    def _clean_content(self, content: str) -> str:
        """Clean content for better LLM processing."""
        if not content:
            return ""
        
        # Replace newlines with spaces first
        content = re.sub(r'\n+', ' ', content)
        # Remove excessive whitespace
        content = re.sub(r'[ \t]+', ' ', content)
        
        return content.strip()
    
    def create_citations(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Create citation data compatible with the existing citation system."""
        citations = []
        for i, result in enumerate(results, 1):
            citation = {
                "label": f"[{i}]",
                "segments": [{
                    "label": f"{i}",
                    "short_url": f"[{i}]",
                    "value": result.url,
                    "title": result.title,
                    "content": result.snippet
                }]
            }
            citations.append(citation)
        
        return citations


class FirecrawlSearchTool:
    """Search tool using Firecrawl API for web scraping and content extraction."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.firecrawl.dev"):
        """Initialize Firecrawl search tool.
        
        Args:
            api_key: Firecrawl API key
            base_url: Firecrawl API base URL
        """
        if not api_key:
            raise ValueError("Firecrawl API key cannot be empty")
            
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def search_and_scrape(self, query: str, max_results: int = 5, max_content_length: int = 4000) -> List[SearchResult]:
        """Search for URLs and scrape content using Firecrawl.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            max_content_length: Maximum content length per result
            
        Returns:
            List of SearchResult objects
        """
        # First, get search results from multiple sources
        search_urls = self._get_search_urls(query, max_results)
        
        # Then scrape each URL using Firecrawl
        results = []
        for url in search_urls:
            try:
                content = self._scrape_url(url, max_content_length)
                if content and content.get('content'):  # Only add if we got meaningful content
                    results.append(SearchResult(
                        title=content.get('title', 'No Title'),
                        url=url,
                        content=content.get('content', ''),
                        snippet=content.get('description', content.get('content', '')[:200] + '...'),
                        source='firecrawl'
                    ))
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                continue
                
            # Stop once we have enough results
            if len(results) >= max_results:
                break
        
        return results
    
    def _get_search_urls(self, query: str, max_results: int) -> List[str]:
        """Get search URLs using multiple strategies with fallbacks."""
        urls = []
        
        # Strategy 1: Try Firecrawl search first
        try:
            firecrawl_urls = self._firecrawl_search(query, max_results)
            urls.extend(firecrawl_urls)
        except Exception as e:
            print(f"Firecrawl search failed: {str(e)}")
        
        # Strategy 2: If we don't have enough URLs, try DuckDuckGo
        if len(urls) < max_results:
            try:
                duckduckgo_urls = self._duckduckgo_search(query, max_results - len(urls))
                urls.extend(duckduckgo_urls)
            except Exception as e:
                print(f"DuckDuckGo search failed: {str(e)}")
        
        # Strategy 3: Final fallback - construct URLs for common sites
        if len(urls) < max_results:
            constructed_urls = self._construct_search_urls(query, max_results - len(urls))
            urls.extend(constructed_urls)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls[:max_results]
    
    def _firecrawl_search(self, query: str, max_results: int) -> List[str]:
        """Search using Firecrawl's search endpoint."""
        response = self.session.post(
            f"{self.base_url}/v1/search",
            json={
                "query": query,
                "limit": max_results,
                "country": "us",
                "lang": "en"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('data'):
                return [result.get('url') for result in data['data'] if result.get('url')]
        else:
            raise Exception(f"Firecrawl search returned status {response.status_code}")
        
        return []
    
    def _duckduckgo_search(self, query: str, max_results: int) -> List[str]:
        """Enhanced DuckDuckGo search with better URL extraction."""
        try:
            # Use DuckDuckGo Instant Answer API
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={
                    'q': query,
                    'format': 'json',
                    'no_html': 1,
                    'skip_disambig': 1
                },
                timeout=10
            )
            
            urls = []
            if response.status_code == 200:
                data = response.json()
                
                # Extract URLs from various DuckDuckGo response fields
                if data.get('AbstractURL'):
                    urls.append(data['AbstractURL'])
                
                if data.get('Results'):
                    for result in data['Results'][:max_results]:
                        if result.get('FirstURL'):
                            urls.append(result['FirstURL'])
                
                if data.get('RelatedTopics'):
                    for topic in data['RelatedTopics'][:max_results]:
                        if isinstance(topic, dict) and topic.get('FirstURL'):
                            urls.append(topic['FirstURL'])
            
            return urls[:max_results]
            
        except Exception as e:
            print(f"DuckDuckGo search error: {str(e)}")
            return []
    
    def _construct_search_urls(self, query: str, max_results: int) -> List[str]:
        """Construct URLs for common authoritative sites as final fallback."""
        encoded_query = query.replace(' ', '+')
        fallback_urls = [
            f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            f"https://www.reddit.com/search/?q={encoded_query}",
            f"https://stackoverflow.com/search?q={encoded_query}",
            f"https://news.ycombinator.com/item?id=search&q={encoded_query}",
            f"https://medium.com/search?q={encoded_query}",
        ]
        
        return fallback_urls[:max_results]
    
    def _scrape_url(self, url: str, max_content_length: int) -> Optional[Dict[str, Any]]:
        """Scrape content from a URL using Firecrawl with enhanced error handling."""
        try:
            response = self.session.post(
                f"{self.base_url}/v1/scrape",
                json={
                    "url": url,
                    "formats": ["markdown", "html"],
                    "includeTags": ["title", "meta", "h1", "h2", "h3", "p", "article"],
                    "excludeTags": ["script", "style", "nav", "footer", "aside", "advertisement"],
                    "onlyMainContent": True,
                    "timeout": 30000
                },
                timeout=45  # Allow extra time for scraping
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('data'):
                    content_data = data['data']
                    
                    # Extract content, preferring markdown
                    content = content_data.get('markdown', content_data.get('content', ''))
                    
                    # Clean and validate content
                    if not content or len(content.strip()) < 50:
                        # Content too short, might be an error page
                        return None
                    
                    # Truncate content if too long
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + '...'
                    
                    return {
                        'title': content_data.get('title', 'No Title'),
                        'content': content,
                        'description': content_data.get('description', ''),
                        'url': url,
                        'metadata': content_data.get('metadata', {})
                    }
            else:
                print(f"Failed to scrape {url}: HTTP {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Timeout scraping {url}")
            return None
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for LLM consumption with improved structure."""
        if not results:
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            # Ensure content is clean and readable
            clean_content = self._clean_content(result.content)
            
            formatted_result = f"""
**Source {i}: {result.title}**
URL: {result.url}
Content: {clean_content}

---
"""
            formatted_results.append(formatted_result)
        
        return "\n".join(formatted_results)
    
    def _clean_content(self, content: str) -> str:
        """Clean content for better LLM processing."""
        if not content:
            return ""
        
        # Remove excessive whitespace and newlines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove markdown artifacts that might confuse the LLM
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Remove markdown links
        
        return content.strip()
    
    def create_citations(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Create citation data compatible with the existing citation system.
        
        This attempts to mimic Google's grounding metadata structure as closely as possible.
        """
        citations = []
        for i, result in enumerate(results, 1):
            citation = {
                "label": f"[{i}]",
                "segments": [{
                    "label": f"{i}",
                    "short_url": f"[{i}]",
                    "value": result.url,
                    "title": result.title,
                    "content": result.snippet
                }]
            }
            citations.append(citation)
        
        return citations
    
    def create_enhanced_citations(self, results: List[SearchResult], response_text: str) -> List[Dict[str, Any]]:
        """Create enhanced citations that more closely match Google's grounding system.
        
        This method attempts to find where sources are actually referenced in the response text
        to create more precise citation markers, similar to Google's grounding metadata.
        """
        citations = []
        
        for i, result in enumerate(results, 1):
            # Try to find references to this source in the response
            citation_positions = self._find_citation_positions(response_text, result.title, result.content)
            
            if citation_positions:
                for start_idx, end_idx in citation_positions:
                    citation = {
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "segments": [{
                            "label": f"{i}",
                            "short_url": f"[{i}]",
                            "value": result.url,
                            "title": result.title,
                            "content": result.snippet
                        }]
                    }
                    citations.append(citation)
            else:
                # Fallback: create a citation without specific positioning
                citation = {
                    "label": f"[{i}]",
                    "segments": [{
                        "label": f"{i}",
                        "short_url": f"[{i}]",
                        "value": result.url,
                        "title": result.title,
                        "content": result.snippet
                    }]
                }
                citations.append(citation)
        
        return citations
    
    def _find_citation_positions(self, text: str, title: str, content: str) -> List[tuple]:
        """Find positions in text where this source might be referenced."""
        positions = []
        
        # Look for mentions of key terms from the title or content
        search_terms = []
        
        # Extract key terms from title
        title_words = [word.strip('.,!?";:') for word in title.split() if len(word) > 3]
        search_terms.extend(title_words[:3])  # Top 3 words from title
        
        # Extract key phrases from content
        sentences = content.split('.')[:2]  # First two sentences
        for sentence in sentences:
            words = [word.strip('.,!?";:') for word in sentence.split() if len(word) > 4]
            search_terms.extend(words[:2])  # Top 2 words per sentence
        
        # Find positions of these terms in the response text
        for term in search_terms:
            for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                start = match.start()
                end = match.end()
                positions.append((start, end))
        
        # Remove duplicates and sort
        positions = list(set(positions))
        positions.sort()
        
        return positions[:3]  # Limit to first 3 matches


def create_search_tool(search_tool_type: str, **kwargs):
    """Factory function to create search tools with enhanced validation.
    
    Args:
        search_tool_type: Type of search tool ('firecrawl', 'brave')
        **kwargs: Additional arguments for the search tool
        
    Returns:
        Search tool instance or None if not supported
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    if search_tool_type == "firecrawl":
        api_key = kwargs.get('api_key')
        base_url = kwargs.get('base_url', 'https://api.firecrawl.dev')
        
        if not api_key:
            raise ValueError("Firecrawl API key is required")
        
        try:
            return FirecrawlSearchTool(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise ValueError(f"Failed to initialize Firecrawl search tool: {str(e)}")
    
    elif search_tool_type == "brave":
        api_key = kwargs.get('api_key')
        base_url = kwargs.get('base_url', 'https://api.search.brave.com')
        
        if not api_key:
            raise ValueError("Brave Search API key is required")
        
        try:
            return BraveSearchTool(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise ValueError(f"Failed to initialize Brave search tool: {str(e)}")
    
    elif search_tool_type == "google":
        # Future: Could implement Google Custom Search API
        raise NotImplementedError("Google Custom Search API not yet implemented")
    
    else:
        raise ValueError(f"Unsupported search tool type: {search_tool_type}")