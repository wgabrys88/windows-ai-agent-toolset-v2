import ctypes, json, time, base64, urllib.request, struct, zlib, math, logging, sys

def load_scenario(filename, scenario_num):
    """Load a specific scenario from the test scenarios file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract shared system prompt
        shared_system_prompt = None
        if '=== SHARED_SYSTEM_PROMPT ===' in content:
            parts = content.split('=== SHARED_SYSTEM_PROMPT ===')
            if len(parts) > 1:
                shared_block = parts[1].split('=== SCENARIO')[0]
                shared_system_prompt = shared_block.strip()
        
        scenarios = content.split('=== SCENARIO ')
        if scenario_num < 1 or scenario_num > len(scenarios) - 1:
            log.error("Invalid scenario number %d. Available: 1-%d", scenario_num, len(scenarios) - 1)
            sys.exit(1)
        
        scenario_text = scenarios[scenario_num]
        lines = scenario_text.strip().split('\n')
        
        # Parse scenario
        system_prompt = shared_system_prompt  # Use shared by default
        task_prompt = None
        tools_schema = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('SYSTEM_PROMPT:'):
                # Override shared if scenario has its own
                system_prompt = line[14:].strip()
            elif line.startswith('TASK_PROMPT:'):
                task_prompt = line[12:].strip()
            elif line.startswith('TOOLS_SCHEMA:'):
                tools_schema_str = line[13:].strip()
                tools_schema = json.loads(tools_schema_str)
            i += 1
        
        if not system_prompt or not task_prompt or not tools_schema:
            log.error("Failed to parse scenario %d", scenario_num)
            sys.exit(1)
        
        log.info("Loaded scenario %d", scenario_num)
        return system_prompt, task_prompt, tools_schema
    
    except FileNotFoundError:
        log.error("Scenario file not found: %s", filename)
        sys.exit(1)
    except Exception as e:
        log.error("Error loading scenario: %s", e)
        sys.exit(1)


LM_STUDIO_ENDPOINT = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "qwen/qwen3-vl-2b-instruct"
TIMEOUT = 240
MAX_STEPS = 60
STEP_DELAY = 0.4
TEMPERATURE = 0.2
MAX_TOKENS = 2048

DUMP_SCREENSHOTS = True
DUMP_PREFIX = "dump_screen_"
DUMP_START = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")

from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

if not hasattr(wintypes, "HCURSOR"):
    wintypes.HCURSOR = wintypes.HANDLE
if not hasattr(wintypes, "HBITMAP"):
    wintypes.HBITMAP = wintypes.HANDLE
if not hasattr(wintypes, "HICON"):
    wintypes.HICON = wintypes.HANDLE

user32.SetProcessDpiAwarenessContext.argtypes = [wintypes.HANDLE]
user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL

user32.GetDesktopWindow.argtypes = []
user32.GetDesktopWindow.restype = wintypes.HWND

user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC

user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = wintypes.INT

user32.SetCursorPos.argtypes = [wintypes.INT, wintypes.INT]
user32.SetCursorPos.restype = wintypes.BOOL

user32.mouse_event.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
user32.mouse_event.restype = None

gdi32.GetDeviceCaps.argtypes = [wintypes.HDC, wintypes.INT]
gdi32.GetDeviceCaps.restype = wintypes.INT

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC

gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL

gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ

gdi32.BitBlt.argtypes = [wintypes.HDC, wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT, wintypes.HDC, wintypes.INT, wintypes.INT, wintypes.DWORD]
gdi32.BitBlt.restype = wintypes.BOOL

gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL

# Cursor-related structures and functions
class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HCURSOR),
        ("ptScreenPos", POINT),
    ]

class ICONINFO(ctypes.Structure):
    _fields_ = [
        ("fIcon", wintypes.BOOL),
        ("xHotspot", wintypes.DWORD),
        ("yHotspot", wintypes.DWORD),
        ("hbmMask", wintypes.HBITMAP),
        ("hbmColor", wintypes.HBITMAP),
    ]

user32.GetCursorInfo.argtypes = [ctypes.POINTER(CURSORINFO)]
user32.GetCursorInfo.restype = wintypes.BOOL

user32.GetIconInfo.argtypes = [wintypes.HICON, ctypes.POINTER(ICONINFO)]
user32.GetIconInfo.restype = wintypes.BOOL

user32.DrawIconEx.argtypes = [
    wintypes.HDC,      # hdc
    wintypes.INT,      # xLeft
    wintypes.INT,      # yTop
    wintypes.HICON,    # hIcon
    wintypes.INT,      # cxWidth
    wintypes.INT,      # cyHeight
    wintypes.UINT,     # istepIfAniCur
    wintypes.HBRUSH,   # hbrFlickerFreeDraw
    wintypes.UINT,     # diFlags
]
user32.DrawIconEx.restype = wintypes.BOOL

user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = wintypes.BOOL

# Constants for cursor
CURSOR_SHOWING = 0x00000001
DI_NORMAL = 0x0003

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT, ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP

try:
    ULONG_PTR = wintypes.ULONG_PTR
except AttributeError:
    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class INPUT_I(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("ii", INPUT_I)]

user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

MOUSEEVENTF_WHEEL = 0x0800

def _dpi_aware():
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4).value)
    except:
        pass

def _screen_size():
    _dpi_aware()
    hwnd = user32.GetDesktopWindow()
    hdc = user32.GetDC(hwnd)
    w = gdi32.GetDeviceCaps(hdc, 118)
    h = gdi32.GetDeviceCaps(hdc, 117)
    user32.ReleaseDC(hwnd, hdc)
    return int(w), int(h)

def _get_cursor_pos():
    p = POINT()
    if not user32.GetCursorPos(ctypes.byref(p)):
        return 0, 0
    return int(p.x), int(p.y)

def _cursor_pos_norm(sw, sh):
    cx, cy = _get_cursor_pos()
    if sw <= 1 or sh <= 1:
        return cx, cy, 0, 0
    xn = int(round((cx / (sw - 1)) * 1000.0))
    yn = int(round((cy / (sh - 1)) * 1000.0))
    if xn < 0: xn = 0
    if yn < 0: yn = 0
    if xn > 1000: xn = 1000
    if yn > 1000: yn = 1000
    return cx, cy, xn, yn

def _draw_cursor_on_dc(hdc_mem, screen_offset_x=0, screen_offset_y=0):
    """
    Draw the current cursor onto the memory DC.
    Returns True if cursor was drawn, False otherwise.
    """
    cursor_info = CURSORINFO()
    cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
    
    if not user32.GetCursorInfo(ctypes.byref(cursor_info)):
        log.warning("GetCursorInfo failed: %d", ctypes.get_last_error())
        return False
    
    # Check if cursor is visible
    if not (cursor_info.flags & CURSOR_SHOWING):
        log.debug("Cursor not showing")
        return False
    
    # Get icon info to find hotspot
    icon_info = ICONINFO()
    if not user32.GetIconInfo(cursor_info.hCursor, ctypes.byref(icon_info)):
        log.warning("GetIconInfo failed: %d", ctypes.get_last_error())
        return False
    
    # Calculate cursor drawing position (account for hotspot)
    cursor_x = cursor_info.ptScreenPos.x - int(icon_info.xHotspot) - screen_offset_x
    cursor_y = cursor_info.ptScreenPos.y - int(icon_info.yHotspot) - screen_offset_y
    
    # Draw the cursor
    success = user32.DrawIconEx(
        hdc_mem,
        cursor_x,
        cursor_y,
        cursor_info.hCursor,
        0,  # use default width
        0,  # use default height
        0,  # not animated
        None,  # no flicker-free brush
        DI_NORMAL
    )
    
    # Clean up icon bitmaps
    if icon_info.hbmMask:
        gdi32.DeleteObject(icon_info.hbmMask)
    if icon_info.hbmColor:
        gdi32.DeleteObject(icon_info.hbmColor)
    
    if not success:
        log.warning("DrawIconEx failed: %d", ctypes.get_last_error())
        return False
    
    log.debug("Cursor drawn at (%d, %d) with hotspot (%d, %d)", 
              cursor_x, cursor_y, icon_info.xHotspot, icon_info.yHotspot)
    return True

def _capture_bgr24():
    """
    Capture screen with cursor included.
    """
    _dpi_aware()
    hwnd = user32.GetDesktopWindow()
    hdc_screen = user32.GetDC(hwnd)
    w = int(gdi32.GetDeviceCaps(hdc_screen, 118))
    h = int(gdi32.GetDeviceCaps(hdc_screen, 117))
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    bmi = BITMAPINFO()
    ctypes.memset(ctypes.byref(bmi), 0, ctypes.sizeof(bmi))
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 24
    bmi.bmiHeader.biCompression = 0
    stride = ((w * 3 + 3) // 4) * 4
    bmi.bmiHeader.biSizeImage = stride * h
    bits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(hdc_mem, ctypes.byref(bmi), 0, ctypes.byref(bits), 0, 0)
    old = gdi32.SelectObject(hdc_mem, hbmp)
    
    # Capture screen
    gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020)
    
    # Draw cursor on top
    _draw_cursor_on_dc(hdc_mem, 0, 0)
    
    size = stride * h
    buf = (ctypes.c_ubyte * size).from_address(bits.value)
    data = bytes(buf)
    gdi32.SelectObject(hdc_mem, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(hwnd, hdc_screen)
    return data, w, h, stride

def _lanczos(x, a=3.0):
    ax = abs(x)
    if ax >= a:
        return 0.0
    if ax < 1e-9:
        return 1.0
    px = math.pi * x
    return (math.sin(px) / px) * (math.sin(px / a) / (px / a))

def _precompute_weights(src_n, dst_n, a=3.0):
    scale = src_n / dst_n
    table = []
    for i in range(dst_n):
        center = (i + 0.5) * scale - 0.5
        left = int(math.floor(center - a + 1))
        right = int(math.floor(center + a))
        pairs = []
        s = 0.0
        for j in range(left, right + 1):
            if 0 <= j < src_n:
                w = _lanczos(j - center, a)
                if w != 0.0:
                    pairs.append((j, w))
                    s += w
        if s != 0.0:
            pairs = [(j, w / s) for (j, w) in pairs]
        else:
            jj = min(src_n - 1, max(0, int(round(center))))
            pairs = [(jj, 1.0)]
        table.append(pairs)
    return table

_WEIGHTS_X = {}
_WEIGHTS_Y = {}

def _get_weights(src_w, src_h, dst_w, dst_h):
    kx = (src_w, dst_w)
    ky = (src_h, dst_h)
    xw = _WEIGHTS_X.get(kx)
    if xw is None:
        xw = _precompute_weights(src_w, dst_w)
        _WEIGHTS_X[kx] = xw
    yw = _WEIGHTS_Y.get(ky)
    if yw is None:
        yw = _precompute_weights(src_h, dst_h)
        _WEIGHTS_Y[ky] = yw
    return xw, yw

def _resize_bgr24_to_rgb24(bgr, src_w, src_h, src_stride, dst_w=1344, dst_h=756, xw=None, yw=None):
    if xw is None or yw is None:
        xw, yw = _precompute_weights(src_w, dst_w), _precompute_weights(src_h, dst_h)
    inter = bytearray(src_h * dst_w * 3)
    for y in range(src_h):
        row_off = y * src_stride
        out_off = y * dst_w * 3
        for x in range(dst_w):
            r = g = b = 0.0
            for sx, w in xw[x]:
                si = row_off + sx * 3
                bb = bgr[si]
                gg = bgr[si + 1]
                rr = bgr[si + 2]
                b += bb * w
                g += gg * w
                r += rr * w
            di = out_off + x * 3
            inter[di] = 0 if r < 0 else 255 if r > 255 else int(r + 0.5)
            inter[di + 1] = 0 if g < 0 else 255 if g > 255 else int(g + 0.5)
            inter[di + 2] = 0 if b < 0 else 255 if b > 255 else int(b + 0.5)
    out = bytearray(dst_h * dst_w * 3)
    for y in range(dst_h):
        out_off = y * dst_w * 3
        wy = yw[y]
        for x in range(dst_w):
            r = g = b = 0.0
            for sy, w in wy:
                si = (sy * dst_w + x) * 3
                rr = inter[si]
                gg = inter[si + 1]
                bb = inter[si + 2]
                r += rr * w
                g += gg * w
                b += bb * w
            di = out_off + x * 3
            out[di] = 0 if r < 0 else 255 if r > 255 else int(r + 0.5)
            out[di + 1] = 0 if g < 0 else 255 if g > 255 else int(g + 0.5)
            out[di + 2] = 0 if b < 0 else 255 if b > 255 else int(b + 0.5)
    return bytes(out), dst_w, dst_h

def _png_from_rgb24(rgb, w, h):
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = bytearray()
    row = w * 3
    for y in range(h):
        raw.append(0)
        off = y * row
        raw.extend(rgb[off:off + row])
    comp = zlib.compress(bytes(raw), 6)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")

def take_screenshot_png_1344():
    bgr, sw, sh, stride = _capture_bgr24()
    xw, yw = _get_weights(sw, sh, 1344, 756)
    rgb, w, h = _resize_bgr24_to_rgb24(bgr, sw, sh, stride, 1344, 756, xw, yw)
    return _png_from_rgb24(rgb, w, h), sw, sh

def _post(payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(LM_STUDIO_ENDPOINT, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _parse_norm_xy(arg_str):
    try:
        a = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
    except:
        a = {}
    if isinstance(a, dict):
        x = a.get("x", 500)
        y = a.get("y", 500)
    elif isinstance(a, (list, tuple)) and len(a) >= 2:
        x, y = a[0], a[1]
    else:
        x, y = 500, 500
    try:
        x = float(x)
    except:
        x = 500.0
    try:
        y = float(y)
    except:
        y = 500.0
    if x < 0: x = 0.0
    if y < 0: y = 0.0
    if x > 1000: x = 1000.0
    if y > 1000: y = 1000.0
    return x, y

def _parse_text(arg_str):
    try:
        a = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
    except:
        a = {}
    if isinstance(a, dict):
        t = a.get("text", "")
    else:
        t = ""
    if t is None:
        t = ""
    return str(t)

def _norm_to_px(xn, yn, sw, sh):
    x = int(round((xn / 1000.0) * (sw - 1)))
    y = int(round((yn / 1000.0) * (sh - 1)))
    return x, y

def _move_mouse_px(x, y):
    _dpi_aware()
    user32.SetCursorPos(int(x), int(y))

def _click_mouse():
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)

def _type_text(text):
    for ch in text:
        code = ord(ch)
        inp = (INPUT * 2)()
        inp[0].type = INPUT_KEYBOARD
        inp[0].ii.ki.wVk = 0
        inp[0].ii.ki.wScan = code
        inp[0].ii.ki.dwFlags = KEYEVENTF_UNICODE
        inp[0].ii.ki.time = 0
        inp[0].ii.ki.dwExtraInfo = 0
        inp[1].type = INPUT_KEYBOARD
        inp[1].ii.ki.wVk = 0
        inp[1].ii.ki.wScan = code
        inp[1].ii.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
        inp[1].ii.ki.time = 0
        inp[1].ii.ki.dwExtraInfo = 0
        
        sent = user32.SendInput(2, inp, ctypes.sizeof(INPUT))
        if sent != 2:
            log.error("SendInput failed sent=%d err=%d sizeof(INPUT)=%d", sent, ctypes.get_last_error(), ctypes.sizeof(INPUT))
        
        time.sleep(0.01)

def _scroll_down_one_notch():
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, ctypes.c_uint(0xFFFFFFFF & (-120)).value, 0)

SYSTEM_PROMPT = (
    "You control a Windows 11 computer using tool calls only. "
    "Tools: take_screenshot, move_mouse(x,y in 0..1000), click_mouse, type_text(text), scroll_down. "
    "Coordinates are normalized integers 0..1000 relative to the screenshot: (0,0) top-left, (1000,1000) bottom-right. "
    "Screenshots now show the mouse cursor position and shape. After move_mouse/click_mouse/type_text/scroll_down, tool responses include "
    "cursor position in pixels and normalized 0..1000. "
)

TASK_PROMPT = "Take a screenshot, move mouse to the center of the notepad++ window, click, type hello, take another screenshot and if hello is visible then scroll down once."


TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "take_screenshot", "description": "Capture screen and return current view with cursor visible.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "move_mouse", "description": "Move mouse using normalized coordinates 0..1000 relative to the screenshot.", "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {"name": "click_mouse", "description": "Left click at current cursor position.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "type_text", "description": "Type text into the focused control.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "scroll_down", "description": "Scroll down by one notch.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]

def main():
    # Use globals as defaults
    system_prompt = SYSTEM_PROMPT
    task_prompt = TASK_PROMPT
    tools_schema = TOOLS_SCHEMA
    
    # Override with scenario if provided
    if len(sys.argv) == 3:
        scenario_file = sys.argv[1]
        scenario_number = int(sys.argv[2])
        system_prompt, task_prompt, tools_schema = load_scenario(scenario_file, scenario_number)
        log.info("Using scenario %d from %s", scenario_number, scenario_file)
    
    log.info("System prompt: %s", system_prompt[:150] + "..." if len(system_prompt) > 150 else system_prompt)
    log.info("Task prompt: %s", task_prompt)
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task_prompt}]
    sw, sh = _screen_size()
    dump_idx = int(DUMP_START)

    for step in range(MAX_STEPS):
        log.info("=" * 80)
        log.info("STEP %d - Sending request to model", step + 1)
        log.info("-" * 80)
        log.info("Current conversation has %d messages", len(messages))
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                # Handle multipart content (text + image)
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append("TEXT: " + part.get("text", ""))
                    elif part.get("type") == "image_url":
                        parts.append("IMAGE: [BASE64_PNG]")
                log.info("  [%d] %s: %s", i, role.upper(), " | ".join(parts))
            elif isinstance(content, str):
                display_content = content[:200] + "..." if len(content) > 200 else content
                log.info("  [%d] %s: %s", i, role.upper(), display_content)
            else:
                log.info("  [%d] %s: [complex content]", i, role.upper())
            
            # Log tool_call_id if present (for tool response messages)
            if "tool_call_id" in msg:
                log.info("      └─ tool_call_id: %s", msg["tool_call_id"])
        
        log.info("-" * 80)
        
        resp = _post({"model": MODEL_ID, "messages": messages, "tools": tools_schema, "tool_choice": "auto", "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS})
        msg = resp["choices"][0]["message"]
        
        log.info("STEP %d - Received response from model", step + 1)
        log.info("-" * 80)
        
        # Log assistant message content
        if msg.get("content"):
            display_content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            log.info("  ASSISTANT: %s", display_content)
        
        # Log tool calls if present
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            log.info("  TOOL_CALLS: %d requested", len(tool_calls))
            for tc in tool_calls:
                tc_name = tc["function"]["name"]
                tc_args = tc["function"].get("arguments", "{}")
                tc_id = tc["id"]
                # Parse args for readable display
                try:
                    args_dict = json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                    args_display = json.dumps(args_dict)
                except:
                    args_display = tc_args
                log.info("    └─ [%s] %s(%s)", tc_id, tc_name, args_display)
        else:
            log.info("  NO TOOL CALLS - Conversation ending")
        
        log.info("=" * 80)
        
        messages.append(msg)

        if not tool_calls:
            log.info("Done: No tool calls in response")
            break

        for tc in tool_calls:
            name = tc["function"]["name"]
            arg_str = tc["function"].get("arguments", "{}")
            call_id = tc["id"]
            
            log.info("Executing tool: %s", name)

            if name == "take_screenshot":
                png_bytes, sw, sh = take_screenshot_png_1344()
                fn = None
                if DUMP_SCREENSHOTS:
                    fn = DUMP_PREFIX + ("%04d" % dump_idx) + ".png"
                    with open(fn, "wb") as f:
                        f.write(png_bytes)
                    dump_idx += 1
                tool_text = "ok" if fn is None else ("ok file=" + fn)
                log.info("  → Screenshot captured: %dx%d, saved as %s", sw, sh, fn if fn else "not saved")
                b64 = base64.b64encode(png_bytes).decode("ascii")
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": tool_text})
                messages = [m for m in messages if not (m.get("role") == "user" and isinstance(m.get("content"), list))]
                messages.append({"role": "user", "content": [{"type": "text", "text": "screen"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]})

            elif name == "move_mouse":
                xn, yn = _parse_norm_xy(arg_str)
                x, y = _norm_to_px(xn, yn, sw, sh)
                _move_mouse_px(x, y)
                time.sleep(0.06)
                cx, cy, cnx, cny = _cursor_pos_norm(sw, sh)
                log.info("  → Moved mouse to norm(%d,%d) → px(%d,%d), actual cursor: px(%d,%d) norm(%d,%d)", 
                         int(xn), int(yn), x, y, cx, cy, cnx, cny)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": "ok cursor_px=%d,%d cursor_norm=%d,%d" % (cx, cy, cnx, cny)})

            elif name == "click_mouse":
                _click_mouse()
                time.sleep(0.06)
                cx, cy, cnx, cny = _cursor_pos_norm(sw, sh)
                log.info("  → Clicked at cursor position: px(%d,%d) norm(%d,%d)", cx, cy, cnx, cny)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": "ok cursor_px=%d,%d cursor_norm=%d,%d" % (cx, cy, cnx, cny)})

            elif name == "type_text":
                text = _parse_text(arg_str)
                _type_text(text)
                time.sleep(0.06)
                cx, cy, cnx, cny = _cursor_pos_norm(sw, sh)
                log.info("  → Typed text: '%s' at cursor: px(%d,%d) norm(%d,%d)", text, cx, cy, cnx, cny)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": "ok typed=%s cursor_px=%d,%d cursor_norm=%d,%d" % (text, cx, cy, cnx, cny)})

            elif name == "scroll_down":
                _scroll_down_one_notch()
                time.sleep(0.06)
                cx, cy, cnx, cny = _cursor_pos_norm(sw, sh)
                log.info("  → Scrolled down at cursor: px(%d,%d) norm(%d,%d)", cx, cy, cnx, cny)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": "ok cursor_px=%d,%d cursor_norm=%d,%d" % (cx, cy, cnx, cny)})

        time.sleep(STEP_DELAY)


if __name__ == "__main__":
    main()
