def rgba_to_hex(rgba_color):
    r, g, b, _ = rgba_color
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'