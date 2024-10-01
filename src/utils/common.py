from datetime import datetime 


def get_job_name_with_datetime(prefix: str) -> str:
    """
    Gets a job name by adding current time to prefix.
    
    Args:
        prefix: A string of job name prefix.
    Returns:
        A job name.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"{prefix}-{now}".replace("_", "-")
    return job_name