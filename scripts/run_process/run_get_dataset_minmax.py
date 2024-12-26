from scripts.process_modules.process_dataarrays_module import calculate_global_min_max_nc
from pathlib import Path

def main():
    dataset=Path(r"Y:\1NEW_DATA\1data\2interim\TSX aa datacubes\ok")
    glob_min = 0
    glob_max = 0
    cubes = list(dataset.rglob('*.nc'))
    print(f"Found {len(cubes)} cubes")
    print(f'---global min {glob_min}, global max {glob_max}---')
    for cube in cubes:
        lmin, lmax = calculate_global_min_max_nc(cube, 'hh')
        glob_min = min(glob_min, lmin)   
        glob_max = max(glob_max, lmax)
        print(f"Global min: {glob_min}, max: {glob_max}")

    print(f"Global min: {glob_min}, max: {glob_max}")



if __name__ == "__main__":
    main()