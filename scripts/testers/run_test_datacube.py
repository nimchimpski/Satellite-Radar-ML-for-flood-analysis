from pathlib import Path

from scripts.process_modules.process_helpers import dataset_type, print_dataarray_info, open_dataarray

cube = Path(r"Y:\1NEW_DATA\1data\2interim\TSX aa datacubes\ok\dims_op_oc_dfd2_695958835_3\695958835_3_extracted\695958835_3.nc")



dataset = Path(r"Y:\1NEW_DATA\1data\2interim\TSX aa datacubes")
cubes = list(dataset.rglob('*.nc'))
for cube in cubes:
    da = open_dataarray(cube)
    dataset_type(da)
    print_dataarray_info(da)

    # #open the cube and get the min mix values for each band
    # hhmin, hhmax = calculate_global_min_max_nc(cube, 'hh')
    # print(f"hh min: {hhmin}, max: {hhmax}")
    # maskmin, maskmax = calculate_global_min_max_nc(cube, 'mask')
    # print(f"mask min: {maskmin}, max: {maskmax}")