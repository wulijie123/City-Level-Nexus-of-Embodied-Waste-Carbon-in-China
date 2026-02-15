# City-Level-Nexus-of-Embodied-Waste-Carbon-in-China
This repository implements the calculation workflow for the city-level nexus of embodied waste carbon in China, covering the quantitative calculation of waste flow based on Input-Output (IO) tables and the measurement of Coupling Coordination Degree (CCD) for 2015 and 2019. The core logic follows the sequence of "data preprocessing → model calculation → result validation → core indicator measurement".

## Calculation Workflow
### 1. 2015 Calculation
#### Prerequisites
- All input files for 2015 are placed in the root `data/` directory (year-specific organization is recommended).
- The `2015/result` directory is created in advance to store calculation results.

#### Execution Steps
```bash
# Navigate to the 2015 directory first
cd 2015

# Step 1: Calculate 2015 waste flow based on IO tables
python waste_flow_cal.py

# Step 2: Calculate 2015 Coupling Coordination Degree (CCD)
python D_Cal.py
