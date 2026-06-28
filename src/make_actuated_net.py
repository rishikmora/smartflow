"""
Converts corridor.net.xml to corridor_actuated.net.xml:
  1. Changes every tlLogic type="static" -> type="actuated"
  2. Adds minDur/maxDur to green phases (duration >= 10s) so the actuated
     controller can actually vary the phase length based on detector data.
     Yellow/intergreen phases (duration < 10s) are left fixed.

Run once; output is reproducible.
"""
import re
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "data", "corridor.net.xml")
DST  = os.path.join(ROOT, "data", "corridor_actuated.net.xml")

MIN_DUR = 5    # minimum green time (s) before actuated can switch
MAX_DUR = 90   # maximum green time (s) before actuated must switch

with open(SRC, "r", encoding="utf-8") as f:
    content = f.read()

# Step 1: static -> actuated
content = re.sub(r'(<tlLogic\b[^>]*\btype=")static(")', r'\1actuated\2', content)
tls_changed = len(re.findall(r'type="actuated"', content))

# Step 2: add minDur/maxDur to green phases (duration >= 10s)
def patch_phase(m: re.Match) -> str:
    duration = int(m.group(1))
    state    = m.group(2)
    if duration >= 10:
        return f'<phase duration="{duration}" minDur="{MIN_DUR}" maxDur="{MAX_DUR}" state={state}/>'
    return m.group(0)

content, phase_patched = re.subn(
    r'<phase duration="(\d+)" state=(\"[^\"]*\")/>',
    patch_phase,
    content,
)

with open(DST, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Written: {DST}")
print(f"  tlLogic converted to actuated: {tls_changed}")
print(f"  Green phases patched with minDur={MIN_DUR} maxDur={MAX_DUR}: {phase_patched}")
