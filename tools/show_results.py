from rich.console import Console
from rich.table import Column, Table
from rich import print

corruptions = ['original', 'gaussian_noise', 'shot_noise', 'impulse_noise', \
    'defocus_blur', 'glass_blur', 'motion_blur', \
        'zoom_blur', 'snow', 'frost', 'fog', 'brightness', \
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

console = Console()
table = Table(show_header=True, header_style="cyan")
print("[bold green]CIFAR10[/bold green]")
table.add_column('Model')
for corruption in corruptions:
    table.add_column(corruption[:3])
console.print(table)

console = Console()
table = Table(show_header=True, header_style="cyan")
print("[bold green]CIFAR100[/bold green]")
table.add_column('Model')
for corruption in corruptions:
    table.add_column(corruption[:3])
console.print(table)

console = Console()
table = Table(show_header=True, header_style="cyan")
print("[bold green]IMAGENET[/bold green]")
table.add_column('Model')
for corruption in corruptions:
    table.add_column(corruption[:3])
console.print(table)

# table.add_column("Date", style="dim", width=12)
# table.add_column("Title")
# table.add_column("Production Budget", justify="right")
# table.add_column("Box Office", justify="right")

# table.add_row(
#     '1', "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
# )
# table.add_row(
#     "May 25, 2018",
#     "[red]Solo[/red]: A Star Wars Story",
#     "$275,000,000",
#     "$393,151,347",
# )
# table.add_row(
#     "Dec 15, 2017",
#     "Star Wars Ep. VIII: The Last Jedi",
#     "$262,000,000",
#     "[bold]$1,332,539,889[/bold]",
# )