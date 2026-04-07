# Quick fix to use plane fitting strategy
import sys
with open('test_fusion_pipeline.py', 'r') as f:
    content = f.read()

# Replace the fusion initialization line
old_line = 'fusion = DepthFusion(filter_size=5, min_box_size=10, boundary_width=2)'
new_line = 'fusion = DepthFusion(filter_size=5, min_box_size=10, boundary_width=2, fusion_strategy="plane")'

content = content.replace(old_line, new_line)

with open('test_fusion_pipeline.py', 'w') as f:
    f.write(content)

print("Updated to use plane fitting strategy")
