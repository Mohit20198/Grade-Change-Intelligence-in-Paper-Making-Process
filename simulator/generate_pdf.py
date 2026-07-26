import markdown
from xhtml2pdf import pisa
import sys

def convert_md_to_pdf(md_file, pdf_file):
    try:
        # Read Markdown file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_text = f.read()

        # Convert Markdown to HTML with extensions for tables
        html_content = markdown.markdown(md_text, extensions=['tables'])

        # Add some basic styling for professional look
        html_wrapper = f"""
        <html>
        <head>
        <style>
            @page {{
                size: a4 portrait;
                margin: 2cm;
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.5;
                color: #333333;
            }}
            h1, h2, h3 {{
                color: #1f2d3d;
            }}
            h1 {{ font-size: 20pt; border-bottom: 2px solid #1f2d3d; padding-bottom: 5px; }}
            h2 {{ font-size: 16pt; margin-top: 20px; }}
            h3 {{ font-size: 13pt; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #dddddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                font-family: monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border: 1px solid #ddd;
                font-family: monospace;
                white-space: pre-wrap;
            }}
        </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """

        # Write to PDF
        with open(pdf_file, "w+b") as out_pdf:
            pisa_status = pisa.CreatePDF(html_wrapper, dest=out_pdf)
            
        if pisa_status.err:
            print("Error occurred while generating PDF")
        else:
            print(f"Successfully created PDF: {pdf_file}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    md_path = r"C:\Users\nehau\Downloads\solution_architecture_document.md"
    pdf_path = r"C:\Users\nehau\Downloads\solution_architecture_document.pdf"
    convert_md_to_pdf(md_path, pdf_path)
