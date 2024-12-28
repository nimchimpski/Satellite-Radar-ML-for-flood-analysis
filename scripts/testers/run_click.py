import click

@click.command()
# @click.option('--count', default=1, help='Number of greetings.')
@click.option('--job_type', prompt='job type1', help='The type of job.')
def job_selector(job_type):    
    """here are instructions for running teh script.
    a file in a dir
    b file in b dir.
    set k to 1 for x."""
    while job_type not in ['train', 'reproduce', 'test', 'debug']:
        print('Invalid job type. Please select from: train, reproduce, test, debug')
        job_type = click.prompt('jobx type2', type=str,  show_default=False)

    click.echo(f"job type is  {job_type}")

if __name__ == '__main__':
    job_selector()