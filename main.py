import argparse

import numpy as np

from datasets.hc18 import VD_HC18
# from datasets.fetalplanesdb import FetalPlanesDB
# from datasets.fetalplanesafrica import FetalPlanesAfrica
# from datasets.fetalabdominal import FetalAbdominal
# from datasets.fetalpsfh import FetalPSFH
# from datasets.fetalacouslic import FetalACOUSLIC
from datasets.fects import FetalEchoCardiography

#from utils import reset_dataset_files, plot_pie_chart, dataset_split_analysis

from torchvision.transforms import Compose, ToTensor, Resize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetal Ultrasound Foundation Model")
    parser.add_argument("--dataset", type=str, required=True, help="path to the dataset")
    parser.add_argument("--reset", action="store_true", help="reset the project")
    args = parser.parse_args()
    
    # ================================================================================== 
    # ==================================== SETTINGS ====================================
    # ================================================================================== 


    # ================================================================================== 
    # =================================== DATASETS =====================================
    # ================================================================================== 

    # reset dataset files
    # if args.reset:
    #     reset_dataset_files(dataset_path=args.dataset)

    # transformations
    # transforms = Compose([Resize((224, 224)), ToTensor()])
    
    # VD_HC18
    VD_HC18_train = VD_HC18(root=args.dataset, split="train", val_percentage=0.1)
    VD_HC18_val = VD_HC18(root=args.dataset, split="val", val_percentage=0.1)
    VD_HC18_test = VD_HC18(root=args.dataset, split="test", val_percentage=0.1)
    
    # FetalPlanesDB (Fetal_Planes_DB_Z3904280)
    # FPDB_train = FetalPlanesDB(
    #     root=args.dataset,
    #     split="train",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )
    # FPDB_val = FetalPlanesDB(
    #     root=args.dataset,
    #     split="val",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )
    # FPDB_test = FetalPlanesDB(
    #     root=args.dataset,
    #     split="test",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )   

    # FetalPlanesAfrica (Fetal_Planes_Africa_Z7540448)
    # FPAfrica_train = FetalPlanesAfrica(
    #     root=args.dataset,
    #     split="train",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )
    # FPAfrica_val = FetalPlanesAfrica(
    #     root=args.dataset,
    #     split="val",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )
    # FPAfrica_test = FetalPlanesAfrica(
    #     root=args.dataset,
    #     split="test",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )   
    
    # FetalAbdominal (Fetal_Abdominal_MD4GCPM9DSC3)
    # FPAbdominal_train = FetalAbdominal(
    #     root=args.dataset,
    #     split="train",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )
    # FPAbdominal_train = FetalAbdominal(
    #     root=args.dataset,
    #     split="val",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )
    # FPAbdominal_test = FetalAbdominal(
    #     root=args.dataset,
    #     split="test",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # ) 
    
    # FetalPSFH (Fetal_PSFH_Z7851339)
    # FPSFH_train = FetalPSFH(
    #     root=args.dataset,
    #     split="train",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )
    # FPSFH_val = FetalPSFH(
    #     root=args.dataset,
    #     split="val",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )
    # FPSFH_test = FetalPSFH(
    #     root=args.dataset,
    #     split="test",
    #     val_percentage=0.1,
    #     target_transform=np.array,
    #     transform=transforms,
    # )

    # FetalACOUSLIC (Fetal_ACOUSLIC_Z12697994)
    # FACOUSLIC_train = FetalACOUSLIC(
    #     root=args.dataset,
    #     split="train",
    #     target_transform=np.array,
    #     transform=transforms,
    # )

    # FECST (Fetal Echocardiography Second Trimester)
    # FECST_train = FetalEchoCardiography(
    #     root=args.dataset,
    #     split="train",
    #     val_size=1,
    #     test_size=2,
    #     target_transform=np.array,
    #     transform=transforms,
    # )