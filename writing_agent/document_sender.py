import anthropic
import base64
import os
import logging
from typing import List, Optional, Dict, Any
import glob
import pathlib
import pypdf

class DocumentSender:
    """
    A class to send document files to Claude.
    For PDFs, it will extract text and send as text blocks.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the document sender.
        
        Args:
            api_key: Optional API key for Anthropic
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            self.logger.warning("No ANTHROPIC_API_KEY provided. Document sending features will not work.")
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        if self.client:
            self.logger.info("Successfully initialized Anthropic client")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        self.logger.info(f"Starting PDF text extraction from: {pdf_path}")
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                self.logger.info(f"PDF has {len(reader.pages)} pages")
                for page_num in range(len(reader.pages)):
                    self.logger.info(f"Processing page {page_num + 1}")
                    page = reader.pages[page_num]
                    # Extract text and normalize spacing
                    page_text = page.extract_text()
                    # Replace multiple spaces and newlines with single space
                    page_text = ' '.join(page_text.split())
                    # Add back paragraph breaks
                    page_text = page_text.replace(". ", ".\n\n")
                    text += page_text + "\n\n"
            
            if text:
                self.logger.info(f"Successfully extracted {len(text)} characters from PDF")
            else:
                self.logger.warning(f"No text content extracted from PDF: {pdf_path}")
            
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""
    
    def encode_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document file for sending to Claude.
        For PDFs, extracts text.
        For images, directly encodes them as base64.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of content objects ready for Claude
        """
        self.logger.info(f"Starting document encoding for: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return []
            
        # Get file extension
        file_ext = pathlib.Path(file_path).suffix.lower()
        self.logger.info(f"Processing file with extension: {file_ext}")
        
        try:
            if file_ext == '.pdf':
                # For PDFs, extract text
                text = self.extract_text_from_pdf(file_path)
                if not text:
                    self.logger.warning(f"No text extracted from PDF: {file_path}")
                    return []
                
                # Split text into manageable chunks (max ~8k chars per chunk)
                MAX_CHUNK_SIZE = 8000
                chunks = []
                for i in range(0, len(text), MAX_CHUNK_SIZE):
                    chunk = text[i:i+MAX_CHUNK_SIZE]
                    chunks.append({
                        "type": "text",
                        "text": f"Document: {os.path.basename(file_path)} (Part {i//MAX_CHUNK_SIZE + 1})\n\n{chunk}"
                    })
                
                self.logger.info(f"Split PDF text into {len(chunks)} chunks for Claude")
                return chunks
                
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                self.logger.info("Processing image file")
                try:
                    with open(file_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    media_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif'
                    }.get(file_ext, 'image/jpeg')
                    
                    self.logger.info(f"Successfully encoded image as {media_type}")
                    return [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data
                        }
                    }]
                except Exception as img_error:
                    self.logger.error(f"Error encoding image {file_path}: {img_error}")
                    return []
            else:
                # For text files, read directly
                self.logger.info("Processing text file")
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    self.logger.info(f"Successfully read {len(content)} characters from text file")
                    return [{
                        "type": "text",
                        "text": f"Document: {os.path.basename(file_path)}\n\n{content}"
                    }]
                except UnicodeDecodeError:
                    self.logger.warning(f"File {file_path} is not a text file or has an unsupported encoding. Skipping.")
                    return []
                except Exception as txt_error:
                    self.logger.error(f"Error reading file {file_path}: {txt_error}")
                    return []
                
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    async def send_query_with_documents(self, 
                                  query: str, 
                                  file_paths: List[str],
                                  model: str = "claude-3-opus-20240229",
                                  max_tokens: int = 4096,
                                  output_file: Optional[str] = None) -> Optional[str]:
        """
        Send a query along with documents directly to Claude for style mimicry.
        
        Args:
            query: The text query to send
            file_paths: List of paths to documents to include
            model: Claude model to use
            max_tokens: Maximum tokens for the response
            output_file: Optional path to save the output text
            
        Returns:
            Claude's response text or None if the query fails
        """
        self.logger.info(f"Starting query with {len(file_paths)} documents")
        
        if not self.client:
            self.logger.error("No Anthropic client available. Check API key.")
            return None
            
        try:
            # Create a more focused style mimicry prompt
            style_mimicry_intro = (
                "I'm sending you reference documents that contain writing styles I'd like you to analyze and mimic. "
                "These are REFERENCE DOCUMENTS ONLY - they demonstrate the writing style, formatting, "
                "and tone that I want you to copy exactly. "
                "Do not focus on their specific content. Instead, focus on: "
                "- Sentence structure and length variation "
                "- Vocabulary choices and complexity "
                "- Paragraph organization and transitions "
                "- Overall tone (formal/informal, academic/conversational) "
                "- Any unique stylistic elements (metaphors, humor, etc.) "
                "\n\nIMPORTANT: Use normal text spacing - do not copy any unusual spacing between words. "
                "After these reference documents, I'll provide content instructions. "
                "MIMIC THE EXACT STYLE of these documents when responding to my instructions, but maintain standard text formatting."
            )
            
            # Start with the style mimicry intro as the first content item
            content = [{
                "type": "text",
                "text": style_mimicry_intro
            }]
            
            # Add documents
            doc_count = 0
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    self.logger.warning(f"File not found: {file_path}")
                    continue
                    
                self.logger.info(f"Processing document: {file_path}")
                doc_content = self.encode_document(file_path)
                if doc_content:
                    # Normalize spacing in document content
                    normalized_content = []
                    for chunk in doc_content:
                        if chunk["type"] == "text":
                            # Replace multiple spaces with single space and normalize newlines
                            text = ' '.join(chunk["text"].split())
                            chunk["text"] = text
                        normalized_content.append(chunk)
                    content.extend(normalized_content)
                    doc_count += 1
                    self.logger.info(f"Successfully added document {file_path} with {len(doc_content)} chunks")
                else:
                    self.logger.warning(f"No content extracted from {file_path}")
            
            # Check if we have any documents
            if doc_count == 0:
                self.logger.warning("No documents were successfully processed")
                # Proceed without style mimicry
                content = [{
                    "type": "text",
                    "text": query
                }]
                self.logger.info("Proceeding with query without style mimicry")
            else:
                # Add the actual query after documents
                self.logger.info("Adding query to content")
                content.append({
                    "type": "text",
                    "text": (
                        "Now, write content following these specifications, but using EXACTLY the "
                        "same writing style as the reference documents above:\n\n" + query
                    )
                })
                
            # Send to Claude
            self.logger.info(f"Sending request to Claude with {len(content)} content blocks")
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
            )
            
            # Combine all text blocks in the response
            response_text = ""
            for content_block in message.content:
                if content_block.type == "text":
                    response_text += content_block.text
            
            self.logger.info(f"Received response from Claude ({len(response_text)} characters)")
            
            # Write output to file if path is provided
            if output_file:
                try:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(response_text)
                    self.logger.info(f"Successfully wrote output to {output_file}")
                except Exception as write_error:
                    self.logger.error(f"Error writing output to file {output_file}: {write_error}")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error sending query with documents: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def load_reference_docs(self, dir_path: Optional[str] = None) -> List[str]:
        """
        Load all reference document paths from a directory.
        
        Args:
            dir_path: Path to the directory containing reference documents
                     (defaults to standard reference_docs directory)
            
        Returns:
            List of file paths to reference documents
        """
        # If no path provided, use default reference_docs directory
        if not dir_path:
            # Use the same reference docs directory as WritingAgent
            from writing_agent.writing_agent import WritingAgent
            dir_path = WritingAgent.REFERENCE_DOCS_DIR
            
        file_paths = []
        
        if os.path.exists(dir_path):
            # Get all files in the directory (non-recursively)
            for file_path in glob.glob(os.path.join(dir_path, "*")):
                if os.path.isfile(file_path) and not os.path.basename(file_path).startswith('.'):
                    file_paths.append(file_path)
            
            self.logger.info(f"Found {len(file_paths)} reference documents in {dir_path}")
        else:
            self.logger.warning(f"Reference documents directory not found: {dir_path}")
        
        return file_paths 