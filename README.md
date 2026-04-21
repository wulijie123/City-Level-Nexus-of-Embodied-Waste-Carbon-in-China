# City-Level-Nexus-of-Embodied-Waste-Carbon-in-China
This repository implements the calculation workflow for the city-level nexus of embodied waste carbon in China, covering the quantitative calculation of waste flow based on Input-Output (IO) tables and the measurement of Coupling Coordination Degree (CCD) for 2015 and 2019. The core logic follows the sequence of "data preprocessing → model calculation → result validation → core indicator measurement".


## Project Structure
```plaintext
│
├── 2015/
│  ├── data/                                   # Data directory (raw/processed data for 2015)
│  ├── waste_flow_cal.py                       # Waste flow calculation based on MRIO model
│  ├── D_Cal.py                                # Coupling Coordination Degree (CCD) calculation
│  └── result/                                 # Results directory for 2015
│    ├── result_c/                             # Carbon transfer results
│    ├── result_hw/                            # Hazardous waste transfer results
│    ├── result_isw/                           # Industrial solid waste transfer results
│    └── CCD_result.xlsx                       # Coupling Coordination Degree (CCD) results
│
├── 2019/
│  ├── data/                                   # Data directory (raw/processed data for 2019)
│  ├── 2019_HW_RAS.py                          # Hazardous waste data balancing using RAS method
│  ├── Regression_Model_Construction.py        # Regression model construction
│  ├── Classification-Regression_Model.py      # Industrial solid waste prediction
│  ├── 2019_ISW_RAS.py                         # ISW data imputation and RAS balancing
│  ├── ISW_prediction_validation.py            # ISW prediction and balancing validation
│  ├── waste_flow_cal.py                       # Waste flow calculation based on MRIO model
│  ├── D_Cal.py                                # Coupling Coordination Degree (CCD) calculation
│  └── result/                                 # Results directory for 2019
│    ├── inventory/                            # 2019 waste inventory construction results
│    │  ├── 2019HW_all.csv                     # Balanced HW data for ISW prediction
│    │  ├── 2019_newdata_combined_predictions.csv # 2019 ISW prediction results
│    │  └── 2019_waste_data.csv                # Final 2019 city-level waste inventory
│    ├── result_c/                             # Carbon transfer results
│    ├── result_hw/                            # Hazardous waste transfer results
│    ├── result_isw/                           # Industrial solid waste transfer results
│    └── CCD_result.xlsx                       # Coupling Coordination Degree (CCD) results
```

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
python 2019_HW_RAS.py

# Step 2: Solid waste prediction (execute the two scripts in sequence)
python Regression_Model_Construction.py
python Classification-Regression_Model.py

# Step 3: 2019 industrial solid waste (ISW) data balancing & carbon emission merging 
# Output: 2019_waste_data.csv (stored in 2019/result/inventory) (The final city-level waste inventory for 2019)
python 2019_ISW_RAS.py

# Step 4: Validate ISW prediction results vs. final adjusted data
python ISW_prediction_validation.py

# Step 5: Calculate 2019 waste flow based on IO tables
python waste_flow_cal.py

# Step 6: Calculate 2019 Coupling Coordination Degree (CCD)
python D_Cal.py
