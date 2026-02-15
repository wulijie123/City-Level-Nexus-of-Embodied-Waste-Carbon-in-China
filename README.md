# City-Level-Nexus-of-Embodied-Waste-Carbon-in-China
This repository implements the calculation workflow for the city-level nexus of embodied waste carbon in China, covering the quantitative calculation of waste flow based on Input-Output (IO) tables and the measurement of Coupling Coordination Degree (CCD) for 2015 and 2019. The core logic follows the sequence of "data preprocessing → model calculation → result validation → core indicator measurement".

## Calculation Workflow
### 1. 2015 Calculation

#### Execution Steps
```bash
# Navigate to the 2015 directory first
cd 2015

# Step 1: Calculate 2015 waste flow based on IO tables
python waste_flow_cal.py

# Step 2: Calculate 2015 Coupling Coordination Degree (CCD)
python D_Cal.py
```
### 2. 2019 Calculation

#### Execution Steps
```bash
# Navigate to the 2019 directory first
cd 2019

# Step 1: 2019 hazardous waste (HW) data balancing 
# Output: 2019HW_all.csv (stored in 2019/result/inventory)
python 2019_HW_RAS.py

# Step 2: Solid waste prediction (execute the two scripts in sequence)
# Results are stored in 2019/result/inventory
python Regression_Model_Construction.py
python Classification-Regression_Model.py

# Step 3: 2019 industrial solid waste (ISW) data balancing & carbon emission merging
# Output: 2019_waste_data.csv (stored in 2019/result/inventory)
python 2019_ISW_RAS.py

# Step 4: Validate ISW prediction results vs. final adjusted data
python ISW_prediction_validation.py

# Step 5: Calculate 2019 waste flow based on IO tables
python waste_flow_cal.py

# Step 6: Calculate 2019 Coupling Coordination Degree (CCD)
python D_Cal.py
