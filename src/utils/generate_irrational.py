def generate_irrational(upper_limit = 45, max_value_integer = 10 ** 5):
    """Generates a random irrational number with upper_limit. 
    
    Imports: 
        package: random, math

    Args:
        upper_limit (float, optional): upper limit for irrational. Defaults to 45.
        max_value_integer (int, optional): maximum value of integer to be square rooted. Defaults to 10**5. 
    
    Returns: 
        irrational number between 0 and upper_limit"""
    
    n = random.randint(1, max_value_integer)
    
    while True:
        n_sqrt = math.sqrt(n)
        if not n_sqrt.is_integer():
            return n_sqrt % upper_limit