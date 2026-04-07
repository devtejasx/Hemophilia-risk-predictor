"""
UI/UX Design Verification Script
Check if CSS styling is properly loaded and applied
"""

import streamlit as st
import re

print("\n" + "="*70)
print("UI/UX DESIGN VERIFICATION")
print("="*70)

# Check 1: Verify CSS function exists
print("\n1️⃣ Checking if CSS function exists...")
try:
    from streamlit_app import get_theme_css, COLORS
    css_output = get_theme_css()
    if len(css_output) > 1000:
        print(f"   ✅ CSS function found ({len(css_output)} characters)")
        print(f"   ✅ Theme colors loaded: {list(COLORS.keys())}")
    else:
        print("   ❌ CSS output too small")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check 2: Verify theme toggle code exists
print("\n2️⃣ Checking if theme toggle UI exists...")
try:
    with open("streamlit_app.py", "r") as f:
        content = f.read()
        if "st.radio" in content and "Theme" in content and "☀️ Light" in content:
            print("   ✅ Theme toggle found in code")
        else:
            print("   ❌ Theme toggle code not found")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check 3: Verify st.markdown call
print("\n3️⃣ Checking if CSS is applied via st.markdown...")
try:
    with open("streamlit_app.py", "r") as f:
        content = f.read()
        if "st.markdown(get_theme_css()" in content and "unsafe_allow_html=True" in content:
            print("   ✅ CSS is applied via st.markdown()")
        else:
            print("   ❌ CSS markdown call not found")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check 4: Count animations in CSS
print("\n4️⃣ Checking animations in CSS...")
try:
    from streamlit_app import get_theme_css
    css = get_theme_css()
    animations = re.findall(r'@keyframes\s+\w+', css)
    print(f"   ✅ Found {len(animations)} animations: {', '.join(animations)}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("WHAT TO CHECK IN YOUR BROWSER:")
print("="*70)
print("""
✓ After running `streamlit run streamlit_app.py`:

1. CLEAR CACHE (Important!):
   - Press: Ctrl + Shift + Delete (or Cmd + Shift + Delete on Mac)
   - Clear: Cache, Cookies, Cached images
   - Close the tab

2. REOPEN THE APP:
   - Go to: http://localhost:8501
   - Wait for it to fully load

3. LOOK FOR THESE CHANGES:

   📌 LIGHT THEME (Default):
      • Background: Light gray (#F9FAFB)
      • Cards: Pure white
      • Text: Dark gray (#111827)
      • Primary color: Indigo (#4F46E5)
      • Sidebar: White background

   📌 DARK THEME (Switch via ☀️/🌙 toggle):
      • Background: Very dark blue (#0F172A)
      • Cards: Dark gray (#1E293B)
      • Text: Light gray (#F1F5F9)
      • Primary color: Light indigo (#6366F1)
      • Smooth transition when switching

   ✨ ANIMATIONS TO EXPECT:
      • Cards slide up smoothly when page loads
      • Cards lift up on hover with shadow
      • Colors fade smoothly when theme switches
      • Form elements have glow effect on focus

4. THEME TOGGLE:
   • Look in sidebar (left side)
   • Under "Settings" section
   • Click: ☀️ Light  or  🌙 Dark
   • Should switch colors smoothly (0.4s transition)

5. IF YOU STILL DON'T SEE CHANGES:
   • Hard refresh: Ctrl + Shift + R (Chrome/Edge/Firefox)
   • Try different browser
   • Check browser console (F12) for errors
   • Delete .streamlit folder: rm -r .streamlit
""")

print("="*70)
print("✅ VERIFICATION COMPLETE - Run the app and check your browser!")
print("="*70 + "\n")
