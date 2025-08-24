import re
import html

def clean_markdown(text: str) -> str:
    """
    Clean and format markdown text for safe HTML display.
    Converts markdown to HTML while sanitizing content.
    """
    if not text:
        return ""
    
    # First, escape any existing HTML to prevent XSS
    text = html.escape(text)
    
    # Normalize numbered lists - convert "1." format to proper markdown
    # This handles cases where the LLM generates "1." for each list item
    lines = text.split('\n')
    normalized_lines = []
    in_numbered_list = False
    
    for line in lines:
        line_stripped = line.strip()
        if re.match(r'^1\. ', line_stripped):
            if not in_numbered_list:
                in_numbered_list = True
            # Keep the "1." format as is - we'll handle it in list processing
            normalized_lines.append(line)
        else:
            if in_numbered_list and not line_stripped:
                in_numbered_list = False
            normalized_lines.append(line)
    
    text = '\n'.join(normalized_lines)
    
    # Convert markdown headers
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Convert bold text - improved handling
    # Handle bold text with ** markers
    text = re.sub(r'\*\*([^*]+?)\*\*', r'<strong>\1</strong>', text)
    # Handle bold text with __ markers
    text = re.sub(r'__([^_]+?)__', r'<strong>\1</strong>', text)
    
    # Convert italic text
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)
    
    # Convert code blocks
    text = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    # Convert links (basic)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', text)
    
    # Convert blockquotes
    text = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    
    # Convert horizontal rules
    text = re.sub(r'^---$', r'<hr>', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\*\*$', r'<hr>', text, flags=re.MULTILINE)
    
    # Process lists - completely rewritten for better handling
    lines = text.split('\n')
    result_lines = []
    in_list = False
    list_type = 'ul'
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Check if this line is a list item
        is_list_item = False
        if re.match(r'^\* ', line_stripped):
            is_list_item = True
            list_type = 'ul'
        elif re.match(r'^- ', line_stripped):
            is_list_item = True
            list_type = 'ul'
        elif re.match(r'^\d+\. ', line_stripped):
            is_list_item = True
            list_type = 'ol'
        # Also handle cases where the LLM generates "1." format
        elif re.match(r'^1\. ', line_stripped):
            is_list_item = True
            list_type = 'ol'
        
        if is_list_item:
            # Extract the content after the list marker
            content = re.sub(r'^[\*\-]\s+', '', line_stripped)
            content = re.sub(r'^\d+\.\s+', '', content)
            
            if not in_list:
                result_lines.append(f'<{list_type}>')
                in_list = True
            
            result_lines.append(f'<li>{content}</li>')
        else:
            # Only close the list if we're not in a numbered list and the next line is not a list item
            if in_list:
                # Check if the next line is also a list item (for consecutive numbered lists)
                if i < len(lines) - 1:
                    next_line = lines[i + 1].strip()
                    if not (re.match(r'^\d+\. ', next_line) or re.match(r'^1\. ', next_line) or 
                           re.match(r'^\* ', next_line) or re.match(r'^- ', next_line)):
                        result_lines.append(f'</{list_type}>')
                        in_list = False
                else:
                    result_lines.append(f'</{list_type}>')
                    in_list = False
            result_lines.append(line)
    
    if in_list:
        result_lines.append(f'</{list_type}>')
    
    text = '\n'.join(result_lines)
    
    # Clean up any <br> tags that got inside list items
    text = re.sub(r'<li>([^<]*)<br>([^<]*)</li>', r'<li>\1 \2</li>', text)
    text = re.sub(r'<li>([^<]*)<br><br>([^<]*)</li>', r'<li>\1 \2</li>', text)
    
    # Handle spacing and line breaks - completely rewritten
    lines = text.split('\n')
    result_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip completely empty lines
        if not line:
            continue
        
        # Add the line
        result_lines.append(line)
        
        # Add appropriate spacing based on what comes next
        if i < len(lines) - 1:
            next_line = lines[i + 1].strip()
            if next_line:
                # If next line is a header, add extra space
                if next_line.startswith('<h'):
                    result_lines.append('<br><br>')
                # If next line is a list item, add space
                elif next_line.startswith('<li>'):
                    result_lines.append('<br>')
                # If current line ends with a colon, add space
                elif line.endswith(':'):
                    result_lines.append('<br>')
                # If next line is a blockquote, add space
                elif next_line.startswith('<blockquote>'):
                    result_lines.append('<br><br>')
                # If next line is a horizontal rule, add space
                elif next_line.startswith('<hr>'):
                    result_lines.append('<br><br>')
                # Otherwise, add normal line break
                else:
                    result_lines.append('<br>')
    
    text = '\n'.join(result_lines)
    
    # Clean up excessive spacing
    # Remove multiple consecutive <br> tags
    text = re.sub(r'(<br>){3,}', r'<br><br>', text)
    
    # Clean up spacing around list items
    text = re.sub(r'<br><br><li>', r'<li>', text)
    text = re.sub(r'</li><br><br>', r'</li>', text)
    
    # Clean up spacing around headers
    text = re.sub(r'<br><br><h[1-3]>', r'<h3>', text)
    text = re.sub(r'</h[1-3]><br><br>', r'</h3>', text)
    
    # Add styling classes
    text = text.replace('<h1>', '<h1 class="text-2xl font-bold mb-4 text-green-400">')
    text = text.replace('<h2>', '<h2 class="text-xl font-bold mb-3 text-green-400">')
    text = text.replace('<h3>', '<h3 class="text-lg font-bold mb-2 text-green-400">')
    text = text.replace('<strong>', '<strong class="font-semibold text-green-300">')
    text = text.replace('<em>', '<em class="italic text-gray-300">')
    text = text.replace('<code>', '<code class="bg-gray-800 px-1 py-0.5 rounded text-green-300 font-mono text-sm">')
    text = text.replace('<pre>', '<pre class="bg-gray-800 p-4 rounded-lg overflow-x-auto mb-4">')
    text = text.replace('<blockquote>', '<blockquote class="border-l-4 border-green-500 pl-4 italic text-gray-300 mb-4">')
    text = text.replace('<hr>', '<hr class="border-green-600 my-4">')
    text = text.replace('<ul>', '<ul class="list-disc list-inside mb-4 space-y-1">')
    text = text.replace('<ol>', '<ol class="list-decimal list-inside mb-4 space-y-1">')
    text = text.replace('<li>', '<li class="text-gray-300">')
    
    return text

def sanitize_html(html_content: str) -> str:
    """
    Sanitize HTML content to prevent XSS attacks.
    """
    # Remove potentially dangerous tags and attributes
    dangerous_tags = ['script', 'iframe', 'object', 'embed', 'form', 'input', 'button']
    dangerous_attrs = ['onclick', 'onload', 'onerror', 'onmouseover', 'onfocus', 'onblur']
    
    for tag in dangerous_tags:
        html_content = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(f'<{tag}[^>]*/?>', '', html_content, flags=re.IGNORECASE)
    
    for attr in dangerous_attrs:
        html_content = re.sub(f'{attr}=["\'][^"\']*["\']', '', html_content, flags=re.IGNORECASE)
    
    return html_content

def format_response_for_frontend(text: str) -> str:
    """
    Format response text specifically for frontend display.
    """
    # Clean markdown first
    cleaned = clean_markdown(text)
    
    # Sanitize HTML
    sanitized = sanitize_html(cleaned)
    
    return sanitized