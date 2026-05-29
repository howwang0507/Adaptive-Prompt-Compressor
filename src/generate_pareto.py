import os


def create_pareto_svg(filepath):
    # Data points (Saving Rate %, Success Rate %)
    data = {
        "Baseline (Raw)": (0, 98, "#7f7f7f"),
        "Static Rule": (8.2, 85, "#d62728"),
        "e-Greedy": (12.4, 82, "#2ca02c"),
        "LinUCB (Ours)": (16.0, 88, "#1f77b4"),
    }

    # SVG Canvas dimensions
    width = 800
    height = 600
    padding = 80

    # Scale functions
    def scale_x(val):
        return padding + (val / 20.0) * (width - 2 * padding)

    def scale_y(val):
        # Invert Y axis for SVG (0 is top)
        return height - padding - ((val - 70) / 35.0) * (height - 2 * padding)

    svg_content = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: white; font-family: sans-serif;">
    
    <!-- Title -->
    <text x="{width / 2}" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">Cost-Quality Pareto Frontier</text>

    <!-- Axes -->
    <!-- X-axis -->
    <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333" stroke-width="2" />
    <text x="{width / 2}" y="{height - 20}" font-size="16" text-anchor="middle" fill="#333">Token Saving Rate (%)</text>
    
    <!-- Y-axis -->
    <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333" stroke-width="2" />
    <text x="30" y="{height / 2}" font-size="16" text-anchor="middle" transform="rotate(-90, 30, {height / 2})" fill="#333">Success Rate (%)</text>

    <!-- Grid lines and labels -->
    """

    # X Grid
    for i in range(0, 21, 5):
        x = scale_x(i)
        svg_content += f'<line x1="{x}" y1="{padding}" x2="{x}" y2="{height - padding}" stroke="#eee" stroke-width="1" />\n'
        svg_content += f'<text x="{x}" y="{height - padding + 20}" font-size="12" text-anchor="middle" fill="#666">{i}</text>\n'

    # Y Grid
    for i in range(70, 105, 10):
        y = scale_y(i)
        svg_content += f'<line x1="{padding}" y1="{y}" x2="{width - padding}" y2="{y}" stroke="#eee" stroke-width="1" />\n'
        svg_content += f'<text x="{padding - 10}" y="{y + 5}" font-size="12" text-anchor="end" fill="#666">{i}</text>\n'

    # Draw Pareto Line (connecting LinUCB and Baseline to show the curve)
    x1, y1 = scale_x(0), scale_y(98)
    x2, y2 = scale_x(16.0), scale_y(88)
    svg_content += f'<path d="M {x1} {y1} Q {scale_x(10)} {scale_y(95)} {x2} {y2}" fill="transparent" stroke="#1f77b4" stroke-width="2" stroke-dasharray="5,5" />\n'

    # Data Points
    for name, (sv, sr, color) in data.items():
        cx = scale_x(sv)
        cy = scale_y(sr)
        svg_content += f'<circle cx="{cx}" cy="{cy}" r="8" fill="{color}" stroke="#333" stroke-width="1" />\n'

        # Adjust label position slightly to avoid overlapping the dot
        label_y = cy - 15 if sr > 85 else cy + 20
        svg_content += f'<text x="{cx}" y="{label_y}" font-size="14" font-weight="bold" text-anchor="middle" fill="{color}">{name}</text>\n'

        # Add values below name
        val_y = label_y + 15 if sr > 85 else label_y + 15
        svg_content += f'<text x="{cx}" y="{val_y}" font-size="12" text-anchor="middle" fill="#666">({sv}%, {sr}%)</text>\n'

    svg_content += "</svg>"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)


if __name__ == "__main__":
    assets_dir = "/Users/wmh/Downloads/Adaptive-Prompt-Compressor/assets"
    os.makedirs(assets_dir, exist_ok=True)
    out_path = os.path.join(assets_dir, "figure_3_pareto.svg")
    create_pareto_svg(out_path)
    print(f"Generated {out_path}")
