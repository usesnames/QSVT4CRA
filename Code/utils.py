"""
Utils 
=========================
Script containing functions usefull across the entire experiment
"""
# =============================================================================
# Module attributes
# =============================================================================
# Documentation format
__docformat__ = 'NumPy'
# License type (e.g. 'GPL', 'MIT', 'Proprietary', ...)
__license__ = 'Proprietary'
# Status ('Prototype', 'Development' or 'Pre-production')
__status__ = 'Development'
# Version (0.0.x = 'Prototype', 0.x.x = 'Development', x.x.x = 'Pre-production)
__version__ = '0.1.0'
# Authors (e.g. code writers)
__author__ = ('Antonello Aita <antonello.aita@gmail.com>',
              'Emanuele Dri <emanuele.dri@polito.it>')
# Maintainer
__maintainer__ = ''
# Email of the maintainer
__email__ = ''

# =============================================================================
# Import modules
# =============================================================================
# Import general purpose module(s)

def mapping(decimal_number, lgd, K):
    """_summary_

    Parameters
    ----------
    decimal_number : _type_
        _description_
    lgd : _type_
        _description_
    K : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # trasforma decimal_number in formato binario con almeno K cifre (offsettando con 0 se necessario)
    b = ('{0:0%sb}' % K).format(decimal_number)
    losses = [loss for i, loss in enumerate(lgd[::-1]) if b[i]=='1']
    #print(losses)
    total_loss = sum(losses)
    return total_loss

def bisection_search(objective, target_value, low_level, high_level, low_value=0, high_value=1, sampler=None, 
                     phis=None, rescaling_factor=1):
    """
    Determines the smallest level such that the objective value is still larger than the target
    :param objective: objective function
    :param target: target value
    :param low_level: lowest level to be considered
    :param high_level: highest level to be considered
    :param low_value: value of lowest level (will be evaluated if set to None)
    :param high_value: value of highest level (will be evaluated if set to None)
    :return: dictionary with level, value, num_eval
    """

    # check whether low and high values are given and evaluated them otherwise
    print('-----------------------------------------------------------------------')
    print('start bisection search for target value %.3f' % target_value)
    print('-----------------------------------------------------------------------')
    num_eval = 0   
    
    total_loss = high_level
    resolution = (total_loss/1000)

    # check if low_value already satisfies the condition
    if low_value == target_value:
        return {'level': low_level, 'value': low_value, 'num_eval': num_eval, 'comment': 'success'}

    # check if high_value is above target
    if high_value == target_value:
        return {'level': high_level, 'value': high_value, 'num_eval': num_eval, 'comment': 'success'}

    # perform bisection search until
    print('low_level      low_val    level         value   high_level    high_value')
    print('-----------------------------------------------------------------------')
    while high_level - low_level > resolution:

        level = (high_level + low_level) / 2.0
        num_eval += 1
        value = objective(sampler, level, phis, comparison=False, rf=rescaling_factor)['estimation_processed']
        
        print('%08.3f\t%05.3f\t%08.3f\t%05.3f\t %08.3f\t %05.3f' \
              % (low_level, low_value, level, value, high_level, high_value))

        if value >= target_value:
            high_level = level
            high_value = value
        else:
            low_level = level
            low_value = value

    # return high value after bisection search
    print('-----------------------------------------------------------------------')
    print('finished bisection search')
    print('-----------------------------------------------------------------------')
    return {'level': high_level, 'value': high_value, 'num_eval': num_eval, 'comment': 'success'}

