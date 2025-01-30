from groq import Groq
import os
from dotenv import load_dotenv
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme
from rich.text import Text
from rich.style import Style
from rich.spinner import Spinner
from rich.status import Status
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.prompt import Prompt
from rich.columns import Columns
from datetime import datetime

# Load environment variables
load_dotenv()

# Enhanced custom theme with more sophisticated colors
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "user": "green bold",
    "assistant": "blue bold",
    "thinking": "yellow italic",
    "timestamp": "dim cyan",
    "border": "bright_blue",
    "title": "bold cyan",
    "accent": "magenta",
    "header": "bold cyan underline",
    "subtle": "dim white",
    "highlight": "bold yellow",
    "input_prompt": "green bold",
    "section_title": "bold blue"
})

console = Console(theme=custom_theme)

class GroqChat:
    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.conversation_history = []
        # Add configuration as class attributes for easy modification
        self.config = {
            'temperature': 0.6,
            'max_completion_tokens': 4096,
            'top_p': 0.95,
            'stream': False  # Changed to False as per docs recommendation for reasoning
        }
        
    def extract_think_content(self, text):
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, text, re.DOTALL)
        response = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
        return think_matches, response
        
    def create_reasoning_prompt(self, user_input):
        """Create a structured reasoning prompt"""
        reasoning_template = (
            f"{user_input}\n\n"
            "Please approach this step-by-step:\n"
            "1. First, analyze the question carefully\n"
            "2. Break down your thinking process using <think> tags\n"
            "3. Validate your reasoning\n"
            "4. Provide a clear, final response"
        )
        return reasoning_template
        
    def chat(self, user_input):
        # Format the user input with reasoning structure
        formatted_input = self.create_reasoning_prompt(user_input)
        self.conversation_history.append({"role": "user", "content": formatted_input})
        
        with Status("[bold yellow]🤔 Processing your request...", spinner="point", console=console) as status:
            try:
                completion = self.client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=self.conversation_history,
                    temperature=self.config['temperature'],
                    max_completion_tokens=self.config['max_completion_tokens'],
                    top_p=self.config['top_p'],
                    stream=self.config['stream'],
                    stop=None,
                )
                
                if self.config['stream']:
                    full_response = ""
                    for chunk in completion:
                        content = chunk.choices[0].delta.content or ""
                        full_response += content
                        status.update(f"[bold yellow]Thinking deeply... {len(full_response)} chars[/]")
                else:
                    full_response = completion.choices[0].message.content
                    
            except Exception as e:
                console.print(Panel(
                    f"[error]API Error: {str(e)}[/]\n"
                    "[info]Attempting to recover...[/]",
                    title="[error]Error[/]",
                    border_style="red"
                ))
                return [], "I apologize, but I encountered an error. Please try again."

        thinking, response = self.extract_think_content(full_response)
        self.conversation_history.append({"role": "assistant", "content": full_response})
        
        # Add conversation management
        if len(self.conversation_history) > 10:  # Prevent context window from getting too large
            self.conversation_history = self.conversation_history[-10:]
            
        return thinking, response

def get_timestamp():
    return f"[timestamp]{datetime.now().strftime('%H:%M:%S')}[/]"

def create_fancy_border(text, padding=2):
    width = len(text) + 2 * padding
    border = "═" * width
    return f"╔{border}╗\n║{' ' * padding}{text}{' ' * padding}║\n╚{border}╝"

def display_welcome():
    welcome_text = """[title]🤖 Welcome to GroqChat[/]

[accent]Your Advanced AI Assistant powered by Groq[/]

[dim]📝 Quick Guide:[/]
[subtle]• Type your messages and press Enter to chat
• Use [highlight]/help[/] for commands
• Use [highlight]/clear[/] to reset the conversation
• Type [highlight]quit[/] to exit
• Responses include thinking process and final answer[/]"""

    welcome_panel = Panel(
        Align.center(welcome_text, vertical="middle"),
        box=box.DOUBLE_EDGE,
        border_style="border",
        padding=(2, 4),
        title="[header]✨ GroqChat v1.0 ✨[/]",
        subtitle=f"Session started at {get_timestamp()}",
        width=100
    )
    console.print("\n")
    console.print(Align.center(welcome_panel))
    console.print("\n")

def format_thinking(thoughts):
    formatted_thoughts = []
    for thought in thoughts:
        formatted_thoughts.append(f"[thinking]💭 {thought.strip()}[/]")
    return "\n\n".join(formatted_thoughts)

def display_model_info():
    """Display model configuration information"""
    model_info = """[section_title]Model Configuration[/]
• Model: DeepSeek R1 (Distil-Llama 70B)
• Temperature: 0.6 (Balanced reasoning)
• Max Tokens: 4096
• Top P: 0.95
"""
    return Panel(
        model_info,
        title="[header]Model Info[/]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2)
    )

def main():
    chat = GroqChat()
    console.clear()
    display_welcome()
    console.print(display_model_info())
    
    while True:
        # Cleaner input section with better alignment
        console.print()  # Add some spacing
        user_input = Prompt.ask(
            f"[input_prompt]{get_timestamp()} │ You[/]",
            console=console
        ).strip()
        
        if user_input.lower() == 'quit':
            goodbye_panel = Panel(
                "[info]Thank you for using GroqChat! Have a great day! 👋[/]",
                border_style="cyan",
                box=box.ROUNDED,
                title="[header]Goodbye![/]",
                padding=(1, 2)
            )
            console.print("\n")
            console.print(Align.center(goodbye_panel))
            break
            
        if user_input.lower() == '/clear':
            console.clear()
            display_welcome()
            continue
            
        if user_input.lower() == '/help':
            help_panel = Panel(
                "[subtle]Available Commands:[/]\n"
                "• [highlight]/clear[/] - Clear the conversation\n"
                "• [highlight]/help[/] - Show this help message\n"
                "• [highlight]quit[/] - Exit the application",
                title="[section_title]Help & Commands[/]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2)
            )
            console.print(help_panel)
            continue
            
        console.print()
        
        try:
            thinking, response = chat.chat(user_input)
            
            if thinking:
                think_panel = Panel(
                    format_thinking(thinking),
                    title=f"{get_timestamp()} [thinking]Thinking Process[/]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(1, 2),
                    width=100
                )
                console.print(Align.center(think_panel))
                console.print()
            
            response_panel = Panel(
                Markdown(response),
                title=f"{get_timestamp()} [assistant]Response[/]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2),
                width=100
            )
            console.print(Align.center(response_panel))
            console.print("\n" + "─" * 100 + "\n", style="dim")
            
        except Exception as e:
            error_panel = Panel(
                f"[error]An error occurred: {str(e)}[/]",
                title="[error]Error[/]",
                border_style="red",
                box=box.ROUNDED,
                padding=(1, 2)
            )
            console.print(error_panel)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[error]Session terminated by user.[/]")
    except Exception as e:
        console.print(f"\n[error]An error occurred: {str(e)}[/]")
