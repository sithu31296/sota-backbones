import models
from rich.table import Table
from rich.console import Console
from rich import box

console = Console()

def show_models():
    model_names = models.__all__
    table = Table(title="Supported Models", box=box.MINIMAL, style="bright_black")
    table.add_column("Model Names", style="red")
    table.add_column("Model Variants", style="cyan")

    for name in model_names:
        table.add_row(name, str(list(eval(f'models.{name.lower()}_settings').keys())))
    console.print(table)


if __name__ == '__main__':
    show_models()