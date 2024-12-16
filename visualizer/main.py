import sys
import json
import tkinter as tk
from PIL import Image, ImageTk

MAX_VISIBLE_NODES = 50
VIEWPORT_PADDING = 100


class Network:
    def __init__(self, config: dict) -> None:
        self.layers = config['layers']
        self.loss = config['loss']
        
        self.inputs = self.layers[0]['input_size']
        self.outputs = self.layers[-1]['output_size']
        
        self.hidden = []
        for layer in self.layers:
            if layer['type'] == 'fully_connected' and layer != self.layers[0]:
                self.hidden.append(layer['output_size'])
        self.hidden = self.hidden[:-1]

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

        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_speed = 10

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        self.container = self.canvas.create_rectangle(0, 0, 0, 0, outline='')

        self.viewport = {
            'left': 0,
            'top': 0,
            'right': width,
            'bottom': height
        }

        self.input_sprites = self.create_clustered_sprites("assets/input.png", self.network.inputs)

        self.hidden_sprites = []
        for layer_size in self.network.hidden:
            layer_sprites = self.create_clustered_sprites("assets/hidden.png", layer_size)
            self.hidden_sprites.append(layer_sprites)

        self.output_sprites = self.create_clustered_sprites("assets/output.png", self.network.outputs)

        self.links = []
        self.set_position()
        self.create_links()

        self.bind("<KeyPress-q>", self.quit_app)
        self.bind("<KeyPress-Escape>", self.quit_app)
        self.bind("<Button-4>", self.zoom_canvas)
        self.bind("<Button-5>", self.zoom_canvas)

        self.bind("<KeyPress-h>", lambda e: self.pan_canvas(1, 0))
        self.bind("<KeyPress-l>", lambda e: self.pan_canvas(-1, 0))
        self.bind("<KeyPress-k>", lambda e: self.pan_canvas(0, -1))
        self.bind("<KeyPress-j>", lambda e: self.pan_canvas(0, 1))

    def quit_app(self, event):
        self.quit()

    def zoom_canvas(self, event):
        mouse_x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        mouse_y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()

        old_zoom = self.zoom

        if event.num == 5 or event.delta < 0:
            self.zoom = max(0.1, self.zoom * 0.9)
        if event.num == 4 or event.delta > 0:
            self.zoom = min(5.0, self.zoom * 1.1)

        zoom_factor = self.zoom / old_zoom

        self.canvas.scale("all", mouse_x, mouse_y, zoom_factor, zoom_factor)

        self.set_position()
        self.create_links()

    def pan_canvas(self, dx, dy):
        self.pan_x += dx * self.pan_speed
        self.pan_y += dy * self.pan_speed

        self.canvas.move("all", dx * self.pan_speed, dy * self.pan_speed)

        self.viewport['left'] = -self.pan_x
        self.viewport['top'] = -self.pan_y
        self.viewport['right'] = self.width - self.pan_x
        self.viewport['bottom'] = self.height - self.pan_y

        self.set_position()
        self.create_links()

    def create_clustered_sprites(self, image_path: str, count: int) -> list:
        if count <= MAX_VISIBLE_NODES:
            return [Sprite(self.canvas, image_path) for _ in range(count)]

        visible_count = MAX_VISIBLE_NODES
        sprites = []
        for i in range(visible_count):
            sprite = Sprite(self.canvas, image_path)
            if i == 0:
                sprite.cluster_size = count // visible_count
                sprite.text_id = self.canvas.create_text(
                    -50, -50,
                    text=f"x{sprite.cluster_size}",
                    fill="white"
                )
            sprites.append(sprite)
        return sprites

    def is_in_viewport(self, x: int, y: int) -> bool:
        return (
            self.viewport['left'] - VIEWPORT_PADDING <= x <= self.viewport['right'] + VIEWPORT_PADDING and
            self.viewport['top'] - VIEWPORT_PADDING <= y <= self.viewport['bottom'] + VIEWPORT_PADDING
        )

    def set_position(self):
        layer_spacing = self.width // (len(self.network.hidden) + 2)
        center_x = self.width // 2 - layer_spacing * (len(self.network.hidden) + 1) // 2
        center_y = self.height // 2

        self.viewport['left'] = -self.pan_x
        self.viewport['top'] = -self.pan_y
        self.viewport['right'] = self.width - self.pan_x
        self.viewport['bottom'] = self.height - self.pan_y

        self.position_layer_sprites(
            self.input_sprites,
            center_x + layer_spacing // 2,
            center_y
        )

        for layer_index, layer_sprites in enumerate(self.hidden_sprites):
            self.position_layer_sprites(
                layer_sprites,
                center_x + (layer_index + 1) * layer_spacing,
                center_y
            )

        self.position_layer_sprites(
            self.output_sprites,
            center_x + (len(self.network.hidden) + 1) * layer_spacing,
            center_y
        )

    def position_layer_sprites(self, sprites: list, x: int, y: int):
        if not sprites:
            return
            
        spacing = self.height // (len(sprites) + 1)
        total_height = len(sprites) * spacing
        start_y = y - total_height // 2
        
        for i, sprite in enumerate(sprites):
            if sprite:
                sprite.x = x
                sprite.y = start_y + (i + 1) * spacing
                
                if hasattr(sprite, 'text_id'):
                    self.canvas.coords(sprite.text_id, sprite.x + 20, sprite.y)
                
                if self.is_in_viewport(sprite.x, sprite.y):
                    self.canvas.coords(sprite.sprite_id, sprite.x, sprite.y)
                    sprite.visible = True
                else:
                    sprite.visible = False

    def create_links(self):
        self.links = []
        for start_layer, end_layer in self.get_layer_pairs():
            for start_sprite in start_layer:
                if start_sprite and start_sprite.visible:
                    for end_sprite in end_layer:
                        if end_sprite and end_sprite.visible:
                            self.links.append(Link(self.canvas, start_sprite, end_sprite))

    def get_layer_pairs(self):
        pairs = []
        pairs.append((self.input_sprites, self.hidden_sprites[0]))
        for i in range(len(self.hidden_sprites) - 1):
            pairs.append((self.hidden_sprites[i], self.hidden_sprites[i + 1]))
        pairs.append((self.hidden_sprites[-1], self.output_sprites))
        return pairs

    def update_sprites(self):
        self.clear_canvas()
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
        self.canvas.delete("all")
        self.update_sprites()
        self.after(1000, self.main_loop)

    def clear_canvas(self):
        self.canvas.delete("all")

def validate_config(config: dict) -> bool:
    if 'layers' not in config:
        print("Error: 'layers' field is missing.")
        return False
    if 'loss' not in config:
        print("Error: 'loss' field is missing.")
        return False
    
    layers = config['layers']
    if not isinstance(layers, list):
        print("Error: 'layers' must be a list.")
        return False
    
    for layer in layers:
        if 'type' not in layer:
            print("Error: layer missing 'type' field.")
            return False
            
        if layer['type'] == 'fully_connected':
            if not all(key in layer for key in ['input_size', 'output_size']):
                print("Error: fully_connected layer missing required fields.")
                return False
            if not all(isinstance(layer[key], int) for key in ['input_size', 'output_size']):
                print("Error: input_size and output_size must be integers.")
                return False
                
        elif layer['type'] == 'activation':
            if 'activation' not in layer:
                print("Error: activation layer missing 'activation' field.")
                return False
                
        elif layer['type'] == 'dropout':
            if 'rate' not in layer:
                print("Error: dropout layer missing 'rate' field.")
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

    network = Network(config)
    window = Window(network)
    window.mainloop()

if __name__ == '__main__':
    main()
