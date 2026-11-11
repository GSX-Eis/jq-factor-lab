#!/usr/bin/env python3
"""
JoinQuant API Documentation Extractor

This script uses Playwright to visit the JoinQuant API documentation page
and extract the complete documentation to a local file for reference.

Usage:
    python utils/extract_jq_docs.py

Output:
    - Creates 'jq_api_docs.html' with the raw HTML documentation
    - Creates 'jq_api_docs.md' with cleaned markdown version
    - Creates 'jq_api_summary.txt' with a structured summary
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import re
from playwright.async_api import async_playwright
import html2text

# Ensure we can import from project root
sys.path.append(str(Path(__file__).parent.parent))


class JQDocExtractor:
    """JoinQuant API Documentation Extractor"""
    
    def __init__(self, output_dir: str = "docs"):
        self.base_url = "https://joinquant.com/help/api/help#name:api"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Output files
        self.html_file = self.output_dir / "jq_api_docs.html"
        self.markdown_file = self.output_dir / "jq_api_docs.md"
        self.summary_file = self.output_dir / "jq_api_summary.txt"
        
    async def extract_documentation(self):
        """Main extraction method"""
        print(f"🚀 Starting JoinQuant API documentation extraction...")
        print(f"📄 Target URL: {self.base_url}")
        
        async with async_playwright() as p:
            # Launch browser (use headless=False for debugging)
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            try:
                # Navigate to the documentation page
                print("🔍 Loading JoinQuant API documentation page...")
                await page.goto(self.base_url, wait_until="networkidle", timeout=30000)
                
                # Wait for content to load
                await page.wait_for_timeout(3000)
                
                # Try to find the main content area
                content_selectors = [
                    ".api-help-content",
                    ".help-content", 
                    ".api-doc-content",
                    ".main-content",
                    "#main-content",
                    ".container",
                    "main",
                    ".content"
                ]
                
                main_content = None
                for selector in content_selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            main_content = element
                            print(f"✅ Found content using selector: {selector}")
                            break
                    except:
                        continue
                
                # If no specific content area found, get the body
                if not main_content:
                    main_content = await page.query_selector("body")
                    print("⚠️  Using full body content as fallback")
                
                # Extract HTML content
                if main_content:
                    html_content = await main_content.inner_html()
                    full_html = await page.content()
                else:
                    html_content = await page.content()
                    full_html = html_content
                
                # Save raw HTML
                await self._save_html(full_html, html_content)
                
                # Convert to markdown
                await self._convert_to_markdown(html_content)
                
                # Extract API summary
                await self._extract_api_summary(page)
                
                print("✅ Documentation extraction completed successfully!")
                print(f"📁 Files saved to: {self.output_dir.absolute()}")
                
            except Exception as e:
                print(f"❌ Error during extraction: {str(e)}")
                # Try to save whatever we can get
                try:
                    content = await page.content()
                    await self._save_html(content, content)
                    print("⚠️  Saved partial content due to error")
                except:
                    print("❌ Could not save any content")
                
            finally:
                await browser.close()
    
    async def _save_html(self, full_html: str, content: str):
        """Save HTML content to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JoinQuant API Documentation - Extracted {timestamp}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; }}
        .extraction-info {{ background: #f0f8ff; padding: 15px; margin: 20px 0; border-left: 4px solid #0066cc; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
        code {{ background: #f0f0f0; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="extraction-info">
        <h2>📚 JoinQuant API Documentation</h2>
        <p><strong>Extracted:</strong> {timestamp}</p>
        <p><strong>Source:</strong> <a href="{self.base_url}">{self.base_url}</a></p>
        <p><strong>Purpose:</strong> Local reference for FactorPilot quantitative research</p>
    </div>
    
    <div class="content">
        {content}
    </div>
</body>
</html>"""
        
        with open(self.html_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"💾 Saved HTML documentation: {self.html_file}")
    
    async def _convert_to_markdown(self, html_content: str):
        """Convert HTML to clean markdown"""
        try:
            # Configure html2text converter
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.body_width = 0  # Don't wrap lines
            h.protect_links = True
            
            # Convert to markdown
            markdown_content = h.handle(html_content)
            
            # Clean up the markdown
            markdown_content = self._clean_markdown(markdown_content)
            
            # Add header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"""# JoinQuant API Documentation

**Extracted:** {timestamp}  
**Source:** {self.base_url}  
**Purpose:** Local reference for FactorPilot quantitative research  

---

"""
            
            with open(self.markdown_file, 'w', encoding='utf-8') as f:
                f.write(header + markdown_content)
            
            print(f"📝 Saved Markdown documentation: {self.markdown_file}")
            
        except Exception as e:
            print(f"⚠️  Could not convert to markdown: {e}")
    
    def _clean_markdown(self, content: str) -> str:
        """Clean up markdown content"""
        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Clean up code blocks
        content = re.sub(r'```\s*\n\s*```', '', content)
        
        # Remove empty headers
        content = re.sub(r'^#+\s*$', '', content, flags=re.MULTILINE)
        
        return content.strip()
    
    async def _extract_api_summary(self, page):
        """Extract API function names and create a structured summary"""
        try:
            # Look for API function patterns
            api_functions = []
            
            # Common selectors for API documentation
            api_selectors = [
                "h1, h2, h3, h4",
                ".api-function",
                ".function-name", 
                "code",
                "pre"
            ]
            
            all_text = await page.inner_text("body")
            
            # Extract function-like patterns from text
            function_patterns = [
                r'def\s+(\w+)\s*\(',  # Python function definitions
                r'(\w+)\s*\([^)]*\)\s*:',  # Function calls with colons
                r'api\.(\w+)',  # API method calls
                r'jq\.(\w+)',   # JoinQuant specific functions
                r'g\.(\w+)',    # Global context functions
            ]
            
            functions_found = set()
            for pattern in function_patterns:
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                functions_found.update(matches)
            
            # Create summary
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary = f"""JoinQuant API Documentation Summary
======================================

Extracted: {timestamp}
Source: {self.base_url}
Purpose: Local reference for FactorPilot quantitative research

API Functions Found:
-------------------
"""
            
            if functions_found:
                for func in sorted(functions_found):
                    if len(func) > 2:  # Filter out very short matches
                        summary += f"- {func}\n"
            else:
                summary += "- Could not automatically extract function list\n"
                summary += "- Please review the HTML/Markdown files for complete documentation\n"
            
            summary += f"""
Total Functions: {len(functions_found)}

Files Generated:
---------------
- {self.html_file.name} - Complete HTML documentation
- {self.markdown_file.name} - Markdown version for easy reading
- {self.summary_file.name} - This summary file

Usage Notes:
-----------
- Use these files as local reference when coding with JoinQuant APIs
- HTML file preserves original formatting and styling
- Markdown file is better for text search and reading
- Integrate these APIs into FactorPilot factor calculations

"""
            
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"📋 Saved API summary: {self.summary_file}")
            
        except Exception as e:
            print(f"⚠️  Could not create summary: {e}")


async def main():
    """Main execution function"""
    print("🎯 JoinQuant API Documentation Extractor")
    print("=" * 50)
    
    # Create extractor
    extractor = JQDocExtractor()
    
    # Run extraction
    await extractor.extract_documentation()
    
    print("\n🎉 Extraction complete! Files ready for use in FactorPilot development.")
    print(f"💡 Tip: Use these docs as reference when implementing JQ data sources in your factors.")


if __name__ == "__main__":
    # Install playwright browsers if needed
    print("🔧 Checking Playwright browser installation...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Extraction cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Try running: playwright install chromium") 