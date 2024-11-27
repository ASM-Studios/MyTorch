import sys
import json
import tkinter as tk
from PIL import Image, ImageTk


class Network:
    def __init__(self, inputs: int, hidden: list[int], outputs: int) -> None:
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

    def __str__(self):
        return f"Network(inputs={self.inputs}, hidden={self.hidden}, outputs={self.outputs})"



class Sprite:
    def __init__(self, canvas: tk.Canvas, image_path: str) -> None:
        self.canvas = canvas
        self.image_path = image_path
        self.x = -50
        self.y = -50
        self.image = Image.open(image_path)
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.sprite_id = self.canvas.create_image(self.x, self.y, image=self.photo_image, anchor=tk.CENTER)

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy
        self.canvas.move(self.sprite_id, dx, dy)

    def update(self) -> None:
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.sprite_id, image=self.photo_image)

class Link:
    def __init__(self, canvas: tk.Canvas, start_sprite: Sprite, end_sprite: Sprite, is_arrow: bool = False) -> None:
        self.canvas = canvas
        self.start_sprite = start_sprite
        self.end_sprite = end_sprite
        self.is_arrow = is_arrow
        arrow_option = 'last' if is_arrow else None
        self.line_id = self.canvas.create_line(
            start_sprite.x, start_sprite.y, end_sprite.x, end_sprite.y, fill="white", arrow=arrow_option
        )

    def update(self) -> None:
        self.canvas.coords(
            self.line_id,
            self.start_sprite.x, self.start_sprite.y,
            self.end_sprite.x, self.end_sprite.y
        )


class Window(tk.Tk):
    def __init__(self, network: Network, width: int = 1920, height: int = 1080) -> None:
        super().__init__()
        self.network = network
        self.title("MyTorch GUI")
        self.geometry(f"{width}x{height}")
        self.resizable(False, False)
        self.width = width
        self.height = height

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        self.input_sprites = []
        for _ in range(self.network.inputs):
            try:
                sprite = Sprite(self.canvas, "assets/input.png")
                self.input_sprites.append(sprite)
            except FileNotFoundError:
                print("Error: 'assets/input.png' not found.")
                self.input_sprites.append(None)

        self.hidden_sprites = []
        for layer_size in self.network.hidden:
            layer_sprites = []
            for _ in range(layer_size):
                try:
                    sprite = Sprite(self.canvas, "assets/hidden.png")
                    layer_sprites.append(sprite)
                except FileNotFoundError:
                    print("Error: 'assets/hidden.png' not found.")
                    layer_sprites.append(None)
            self.hidden_sprites.append(layer_sprites)

        self.output_sprites = []
        for _ in range(self.network.outputs):
            try:
                sprite = Sprite(self.canvas, "assets/output.png")
                self.output_sprites.append(sprite)
            except FileNotFoundError:
                print("Error: 'assets/output.png' not found.")
                self.output_sprites.append(None)

        self.links = []
        self.set_position()
        self.create_links()

        self.bind("<KeyPress-q>", self.quit_app)
        self.bind("<KeyPress-Escape>", self.quit_app)

    def quit_app(self, event):
        self.quit()

    def set_position(self):
        layer_spacing = self.width // (len(self.network.hidden) + 2)

        input_spacing = self.height // (self.network.inputs + 1)
        for i, sprite in enumerate(self.input_sprites):
            if sprite:
                sprite.x = layer_spacing // 2
                sprite.y = (i + 1) * input_spacing
                self.canvas.coords(sprite.sprite_id, sprite.x, sprite.y)

        for layer_index, layer_sprites in enumerate(self.hidden_sprites):
            hidden_spacing = self.height // (len(layer_sprites) + 1)
            for node_index, sprite in enumerate(layer_sprites):
                if sprite:
                    sprite.x = (layer_index + 1) * layer_spacing
                    sprite.y = (node_index + 1) * hidden_spacing
                    self.canvas.coords(sprite.sprite_id, sprite.x, sprite.y)

        output_spacing = self.height // (self.network.outputs + 1)
        for i, sprite in enumerate(self.output_sprites):
            if sprite:
                sprite.x = (len(self.network.hidden) + 1) * layer_spacing
                sprite.y = (i + 1) * output_spacing
                self.canvas.coords(sprite.sprite_id, sprite.x, sprite.y)

    def create_links(self):
        for input_sprite in self.input_sprites:
            if input_sprite:
                for hidden_sprite in self.hidden_sprites[0]:
                    if hidden_sprite:
                        self.links.append(Link(self.canvas, input_sprite, hidden_sprite))

        for layer_index in range(len(self.hidden_sprites) - 1):
            for start_sprite in self.hidden_sprites[layer_index]:
                if start_sprite:
                    for end_sprite in self.hidden_sprites[layer_index + 1]:
                        if end_sprite:
                            self.links.append(Link(self.canvas, start_sprite, end_sprite))

        for hidden_sprite in self.hidden_sprites[-1]:
            if hidden_sprite:
                for output_sprite in self.output_sprites:
                    if output_sprite:
                        self.links.append(Link(self.canvas, hidden_sprite, output_sprite))

    def update_sprites(self):
        for sprite in self.input_sprites:
            if sprite:
                sprite.update()

        for layer_sprites in self.hidden_sprites:
            for sprite in layer_sprites:
                if sprite:
                    sprite.update()

        for sprite in self.output_sprites:
            if sprite:
                sprite.update()

        for link in self.links:
            link.update()

    def main_loop(self):
        self.update_sprites()
        self.after(1000, self.main_loop)


def validate_config(config: dict) -> bool:
    if not isinstance(config.get('inputs'), int):
        print("Error: 'inputs' must be an integer.")
        return False
    if not isinstance(config.get('hidden'), list) or not all(isinstance(i, int) for i in config['hidden']):
        print("Error: 'hidden' must be a list of integers.")
        return False
    if not isinstance(config.get('outputs'), int):
        print("Error: 'outputs' must be an integer.")
        return False
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = None

    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {config_file} does not exist.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {config_file} is not a valid JSON file.")
        sys.exit(1)

    if not validate_config(config):
        sys.exit(1)

    network = Network(config['inputs'], config['hidden'], config['outputs'])

    window = Window(network)
    window.mainloop()

if __name__ == '__main__':
    main()
