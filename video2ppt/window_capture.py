"""
Window capture module for cross-platform window selection and capture.
Supports Windows and macOS.
"""
import platform
import subprocess
import tempfile
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np

class WindowCaptureError(Exception):
    """Base exception for window capture errors."""
    pass

class WindowCapture:
    """Cross-platform window capture functionality."""
    
    def __init__(self):
        self.system = platform.system()
        self.window_info = {}
    
    def list_windows(self) -> list[Dict[str, Any]]:
        """List all available windows.
        
        Returns:
            list: List of dictionaries containing window information.
        """
        if self.system == 'Darwin':  # macOS
            return self._list_windows_macos()
        elif self.system == 'Windows':
            return self._list_windows_windows()
        else:
            raise WindowCaptureError(f"Unsupported platform: {self.system}")
    
    def capture_window(self, window_id: Optional[Union[int, str]] = None, 
                      region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture a window or screen region.
        
        Args:
            window_id: Window ID or title. If None, captures the entire screen.
            region: Optional (x, y, width, height) tuple for capturing a specific region.
            
        Returns:
            numpy.ndarray: Captured image in BGR format.
        """
        if self.system == 'Darwin':
            return self._capture_macos(window_id, region)
        elif self.system == 'Windows':
            return self._capture_windows(window_id, region)
        else:
            raise WindowCaptureError(f"Unsupported platform: {self.system}")
    
    def interactive_select_window(self) -> Dict[str, Any]:
        """Interactively select a window.
        
        Returns:
            dict: Selected window information.
        """
        print("Please select a window to monitor...")
        
        if self.system == 'Darwin':
            return self._interactive_select_macos()
        elif self.system == 'Windows':
            return self._interactive_select_windows()
        else:
            raise WindowCaptureError(f"Unsupported platform: {self.system}")
    
    # macOS specific implementations
    def _list_windows_macos(self) -> list[Dict[str, Any]]:
        """List all windows on macOS using AppleScript."""
        script = '''
        global frontApp, frontAppName, windowTitle, windowId
        
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set frontAppName to name of frontApp
            
            set windowList to {}
            repeat with proc in application processes
                set procName to name of proc
                try
                    set windows to windows of proc
                    repeat with w in windows
                        try
                            set windowTitle to name of w
                            set windowId to id of w
                            copy {procName, windowTitle, windowId} to the end of windowList
                        end try
                    end repeat
                end try
            end repeat
            return windowList
        end tell
        '''
        
        try:
            result = subprocess.run(['osascript', '-e', script], 
                                 capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            windows = []
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    break
                windows.append({
                    'app': lines[i],
                    'title': lines[i+1],
                    'id': lines[i+2]
                })
            return windows
        except subprocess.CalledProcessError as e:
            raise WindowCaptureError(f"Failed to list windows: {e.stderr}")
    
    def _capture_macos(self, window_id: Optional[Union[int, str]] = None, 
                      region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture a window or screen region on macOS."""
        import tempfile
        import os
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_file = tmp.name
        
        try:
            # Build the screencapture command
            cmd = ['screencapture', '-x']  # -x: don't play sound
            
            if window_id is not None:
                cmd.extend(['-l', str(window_id)])  # -l: capture window by windowID
            
            if region is not None:
                x, y, w, h = region
                cmd.extend(['-R', f"{x},{y},{w},{h}"])
            
            cmd.append(temp_file)
            
            # Execute the command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Read the captured image
            if os.path.exists(temp_file):
                img = cv2.imread(temp_file)
                if img is not None:
                    return img
                else:
                    raise WindowCaptureError("Failed to read captured image")
            else:
                raise WindowCaptureError("Screenshot file was not created")
                
        except subprocess.CalledProcessError as e:
            raise WindowCaptureError(f"Screencapture failed: {e.stderr}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _interactive_select_macos(self) -> Dict[str, Any]:
        """Interactively select a window on macOS."""
        print("Please click on the window you want to capture...")
        
        # Use AppleScript to get window ID from user click
        script = '''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set frontAppName to name of frontApp
            
            -- Get window under mouse
            set mousePosition to {0, 0}
            set windowUnderMouse to null
            
            try
                set mousePosition to get value of attribute "AXFocusedUIElement" of process frontAppName
                set windowUnderMouse to window 1 of process frontAppName whose subrole is "AXStandardWindow"
                
                set windowTitle to name of windowUnderMouse
                set windowId to id of windowUnderMouse
                
                return {frontAppName, windowTitle, windowId}
            on error errMsg
                return {"Error", "Failed to get window: " & errMsg, ""}
            end try
        end tell
        '''
        
        try:
            result = subprocess.run(['osascript', '-e', script], 
                                 capture_output=True, text=True, check=True)
            app, title, win_id = result.stdout.strip().split('\n')[:3]
            
            return {
                'app': app,
                'title': title,
                'id': win_id
            }
            
        except subprocess.CalledProcessError as e:
            raise WindowCaptureError(f"Failed to select window: {e.stderr}")
    
    # Windows specific implementations
    def _list_windows_windows(self) -> list[Dict[str, Any]]:
        """List all windows on Windows using PowerShell."""
        # Using a simpler PowerShell command to list windows
        script = """
        Add-Type -TypeDefinition @"
        using System;
        using System.Runtime.InteropServices;
        using System.Text;
        
        public class Win32 {
            [DllImport("user32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            public static extern bool IsWindowVisible(IntPtr hWnd);
            
            [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
            public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
            
            [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
            public static extern int GetWindowTextLength(IntPtr hWnd);
            
            [DllImport("user32.dll")]
            public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
            
            [DllImport("user32.dll")]
            public static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);
            
            public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
            
            [DllImport("user32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            public static extern bool IsIconic(IntPtr hWnd);
            
            [DllImport("user32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            public static extern bool IsZoomed(IntPtr hWnd);
        }
"@

        $windowList = New-Object System.Collections.ArrayList
        
        $windowCallback = {
            param($hWnd, $lParam)
            
            if ([Win32]::IsWindowVisible($hWnd)) {
                $length = [Win32]::GetWindowTextLength($hWnd)
                if ($length -gt 0) {
                    $sb = New-Object System.Text.StringBuilder($length + 1)
                    [void][Win32]::GetWindowText($hWnd, $sb, $sb.Capacity)
                    $title = $sb.ToString()
                    
                    if (![string]::IsNullOrEmpty($title)) {
                        $processId = 0
                        [void][Win32]::GetWindowThreadProcessId($hWnd, [ref]$processId)
                        
                        if ($processId -ne 0) {
                            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                            $processName = if ($process) { $process.ProcessName } else { "Unknown" }
                            
                            $window = New-Object PSObject -Property @{
                                Handle = $hWnd
                                Title = $title
                                ProcessId = $processId
                                ProcessName = $processName
                                IsMinimized = [Win32]::IsIconic($hWnd)
                                IsMaximized = [Win32]::IsZoomed($hWnd)
                            }
                            
                            [void]$windowList.Add($window)
                        }
                    }
                }
            }
            
            return $true
        }
        
        # Create delegate and enumerate windows
        $delegate = [Win32+EnumWindowsProc]$windowCallback
        [void][Win32]::EnumWindows($delegate, [IntPtr]::Zero)
        
        # Convert to array of hashtables for JSON serialization
        $result = @()
        foreach ($win in $windowList) {
            $result += @{
                'id' = $win.Handle
                'title' = $win.Title
                'process' = $win.ProcessName
                'pid' = $win.ProcessId
            }
        }
        
        ConvertTo-Json -InputObject $result -Compress
        """
        
        try:
            result = subprocess.run(
                ['powershell', '-NoProfile', '-NonInteractive', 
                 '-ExecutionPolicy', 'Bypass', '-Command', script],
                capture_output=True, 
                text=True, 
                check=True
            )
            
            import json
            windows = json.loads(result.stdout)
            return windows
            
        except subprocess.CalledProcessError as e:
            raise WindowCaptureError(f"Failed to list windows: {e.stderr}")
        except json.JSONDecodeError:
            raise WindowCaptureError("Failed to parse window list")
    
    def _capture_windows(self, window_handle: Optional[Union[int, str]] = None, 
                        region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture a window or screen region on Windows."""
        import ctypes
        from ctypes import wintypes
        
        # Define required Windows types
        user32 = ctypes.WinDLL('user32')
        gdi32 = ctypes.WinDLL('gdi32')
        
        class RECT(ctypes.Structure):
            _fields_ = [
                ('left', ctypes.c_long),
                ('top', ctypes.c_long),
                ('right', ctypes.c_long),
                ('bottom', ctypes.c_long)
            ]
        
        class WINDOWINFO(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.DWORD),
                ('rcWindow', RECT),
                ('rcClient', RECT),
                ('dwStyle', wintypes.DWORD),
                ('dwExStyle', wintypes.DWORD),
                ('dwWindowStatus', wintypes.DWORD),
                ('cxWindowBorders', wintypes.UINT),
                ('cyWindowBorders', wintypes.UINT),
                ('atomWindowType', wintypes.ATOM),
                ('wCreatorVersion', wintypes.WORD)
            ]
        
        # Get window rectangle
        def get_window_rect(hwnd):
            rect = RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            return rect
        
        # Get client rectangle
        def get_client_rect(hwnd):
            rect = RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            return rect
        
        # Convert window handle
        if window_handle is not None:
            if isinstance(window_handle, str):
                try:
                    hwnd = int(window_handle, 0)  # Handle hex strings
                except ValueError:
                    # Try to find window by title
                    hwnd = user32.FindWindowW(None, window_handle)
                    if hwnd == 0:
                        raise WindowCaptureError(f"Window with title '{window_handle}' not found")
            else:
                hwnd = int(window_handle)
        else:
            # Capture entire screen
            hwnd = user32.GetDesktopWindow()
        
        # Get window dimensions
        if region is not None:
            x, y, width, height = region
            rect = RECT()
            rect.left = x
            rect.top = y
            rect.right = x + width
            rect.bottom = y + height
        else:
            if hwnd == user32.GetDesktopWindow():
                # For entire screen
                width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
                height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
                rect = RECT(0, 0, width, height)
            else:
                # For specific window
                rect = get_window_rect(hwnd)
                width = rect.right - rect.left
                height = rect.bottom - rect.top
        
        # Create device contexts
        hwnd_dc = user32.GetWindowDC(hwnd)
        mfc_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        save_dc = gdi32.SaveDC(mfc_dc)
        
        # Create bitmap
        bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        gdi32.SelectObject(mfc_dc, bmp)
        
        # Capture the window
        result = ctypes.windll.user32.PrintWindow(hwnd, mfc_dc, 1)  # PW_CLIENTONLY=1
        
        if not result:
            # Fallback to BitBlt if PrintWindow fails
            result = gdi32.BitBlt(mfc_dc, 0, 0, width, height, hwnd_dc, 0, 0, 0x00CC0020)  # SRCCOPY
        
        # Create bitmap info header
        bmp_info = (ctypes.c_char * 40)()
        bmp_info[0] = 40  # biSize
        ctypes.memmove(ctypes.byref(bmp_info, 4), ctypes.byref(ctypes.c_ulong(width)), 4)  # biWidth
        ctypes.memmove(ctypes.byref(bmp_info, 8), ctypes.byref(ctypes.c_ulong(height)), 4)  # biHeight
        ctypes.memmove(ctypes.byref(bmp_info, 12), ctypes.byref(ctypes.c_ushort(1)), 2)  # biPlanes
        ctypes.memmove(ctypes.byref(bmp_info, 14), ctypes.byref(ctypes.c_ushort(24)), 2)  # biBitCount
        
        # Get the bitmap bits
        bmp_data = (ctypes.c_ubyte * (width * height * 3))()
        gdi32.GetDIBits(mfc_dc, bmp, 0, height, bmp_data, bmp_info, 0)
        
        # Clean up
        gdi32.RestoreDC(mfc_dc, save_dc)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mfc_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)
        
        # Convert to numpy array
        img = np.frombuffer(bmp_data, dtype=np.uint8)
        img = img.reshape((height, width, 3))
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _interactive_select_windows(self) -> Dict[str, Any]:
        """Interactively select a window on Windows."""
        print("Please click on the window you want to capture...")
        
        import ctypes
        from ctypes import wintypes
        
        # Define required Windows types and functions
        user32 = ctypes.WinDLL('user32')
        
        # Constants
        WH_MOUSE_LL = 14
        WM_LBUTTONDOWN = 0x0201
        
        # Structures
        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ('pt', wintypes.POINT),
                ('mouseData', wintypes.DWORD),
                ('flags', wintypes.DWORD),
                ('time', wintypes.DWORD),
                ('dwExtraInfo', wintypes.ULONG_PTR)
            ]
        
        # Callback function type
        HOOKPROC = ctypes.WINFUNCTYPE(wintypes.LPARAM, wintypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        
        # Global variables
        hook_id = None
        selected_hwnd = None
        
        # Callback function for mouse events
        def low_level_mouse_handler(nCode, wParam, lParam):
            nonlocal selected_hwnd
            
            if wParam == WM_LBUTTONDOWN:
                # Get the window under the cursor
                pt = MSLLHOOKSTRUCT.from_address(lParam).pt
                hwnd = user32.WindowFromPoint(pt)
                
                # Get window title
                length = user32.GetWindowTextLengthW(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value
                
                # Get process ID
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                
                # Get process name
                process_name = "Unknown"
                try:
                    import psutil
                    process = psutil.Process(pid.value)
                    process_name = process.name()
                except:
                    pass
                
                selected_hwnd = {
                    'id': hwnd,
                    'title': title,
                    'process': process_name,
                    'pid': pid.value
                }
                
                # Return non-zero to stop the hook
                return 1
            
            # Call the next hook in the chain
            return user32.CallNextHookEx(hook_id, nCode, wParam, lParam)
        
        # Set up the hook
        hook_proc = HOOKPROC(low_level_mouse_handler)
        hook_id = user32.SetWindowsHookExA(WH_MOUSE_LL, hook_proc, None, 0)
        
        if not hook_id:
            raise WindowCaptureError("Failed to set up mouse hook")
        
        print("Click on the window you want to capture...")
        
        # Message pump
        msg = wintypes.MSG()
        while not selected_hwnd and user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        
        # Clean up
        if hook_id:
            user32.UnhookWindowsHookEx(hook_id)
        
        if not selected_hwnd:
            raise WindowCaptureError("No window was selected")
        
        return selected_hwnd

def test_capture():
    """Test function for window capture."""
    try:
        capturer = WindowCapture()
        
        # List available windows
        print("Listing windows...")
        windows = capturer.list_windows()
        print(f"Found {len(windows)} windows:")
        for i, win in enumerate(windows[:5]):  # Show first 5 windows
            print(f"{i+1}. {win.get('title')} (Process: {win.get('process', 'N/A')})")
        
        # Interactive window selection
        print("\nPlease select a window to capture...")
        selected = capturer.interactive_select_window()
        print(f"\nSelected window: {selected.get('title')}")
        
        # Capture the window
        print("Capturing window...")
        img = capturer.capture_window(selected.get('id'))
        
        # Display the captured image
        cv2.imshow("Captured Window", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(test_capture())
