def main(tile_path, normalized_tiles_path):
    print('++++IN PROCESS TILES NEW DIR')   

    tile_path = Path('/xxx')
    normalized_tiles_path = Path('/xxx')
    

    # if normalized_tiles_path.exists():
    #     shutil.rmtree(normalized_tiles_path)
    # Ensure the main parent folder for normalized data exists
    normalized_tiles_path.mkdir(parents=True, exist_ok=True)

    num_events = sum(1 for i in tile_path.rglob('tiles') if i.is_dir())
    print(f"---Found {num_events} event directories")

    done = 0

    for event in tile_path.iterdir():  # Iterate over event directories
        if event.is_dir():
            print(f"---Processing event: {event.name}")

            # Find 'tiles' folders at any level within each event directory

            for tiles_folder in event.rglob('tiles'):
                if tiles_folder.is_dir():
                    # Construct the path for the mirrored 'normalized_tiles' directory
                    relative_path = tiles_folder.relative_to(tile_path)
                    normalized_tiles_folder = normalized_tiles_path / relative_path.parent / 'normalized_minmax_tiles'
                    normalized_tiles_folder.mkdir(parents=True, exist_ok=True)

                    # Normalize each tile in the 'tiles' folder
                    j = 0
                    for tile in tiles_folder.iterdir():
                        if tile.is_file():
                            # print(f"---Normalizing tile {j}")
                            normalise_a_tile(tile, normalized_tiles_folder)
                            print(f"---tile: {j}. Finished {done} of {num_events} events")
                            j += 1
                    

                    done += 1
                print(f"---Processed {done} of {num_events} events")

if __name__ == '__main__':
    main()
