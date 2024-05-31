from isolate_model_result import Model
import time
import numpy as np

model_paras_files = "/hildafs/projects/phy200028p/mgwalker/m2fs/final_mask_MLP_clamped_76c0e105c350c062a69a_iter600.pt"
model = Model(model_paras_files)
        
input_paras=np.array([4321.,4.321,-3.21,-0.21])
start_time=time.time()
specmodel=model(input_paras)
end_time=time.time()
g0=open('/hildafs/projects/phy200028p/mgwalker/scripts/crap.res','w')
g0.write(str(end_time-start_time))
g0.close()

