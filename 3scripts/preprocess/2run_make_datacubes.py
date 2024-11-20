
from pathlib import Path
from ..modules.tiff_folders_to_datacubes import create_event_datacubes
# cd Y:/1NEW_DATA/1data/2interim/dset_DLR_S1S2_bycountry_4326



def main():
    data_root = Path(r"\\cerndata100\AI_Files\Users\AI_Flood_Service\1NEW_DATA\1data\2interim\tests")
    data_name = "dset_DLR_S1S2_bycountry_4326_test"
    source = data_root / data_name
    save_path = data_root / "dset_DLR_S1S2_bycountry_4326_datacubes"

    create_event_datacubes(source, save_path, VERSION="v1")






    return

if __name__ == "__main__":
    main()