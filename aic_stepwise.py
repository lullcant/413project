import statsmodels.formula.api as smf

def stepwise_selection(data, target):
    variate = set(data.columns) 
    variate.remove(target)
    selected = []
    n = data.shape[0]
    current_score, best_new_score = float('inf'), float('inf')

    # Forward step
    while variate:
        aic_with_variate = []
        for candidate in variate: 
            formula = "{}~{}".format(target, "+".join(selected + [candidate]))
            K = formula.count('+')+1 if n/data.shape[1] <40 else 0
            aic = smf.ols(formula=formula, data=data).fit().aic+ 2*K*(K+1)/(n-K-1)
            aic_with_variate.append((aic, candidate))
        aic_with_variate.sort(reverse=True) 
        best_new_score, best_candidate = aic_with_variate.pop()
        if current_score > best_new_score: 
            variate.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
        else:
            break
    
    # Backward step
    while len(selected) > 0:
        aic_with_variate = []
        for candidate in selected: 
            selected_new = selected.copy()
            selected_new.remove(candidate)
            formula = "{}~{}".format(target, "+".join(selected_new))
            K = formula.count('+')+1 if n/data.shape[1] <40 else 0
            aic = smf.ols(formula=formula, data=data).fit().aic+ 2*K*(K+1)/(n-K-1)
            aic_with_variate.append((aic, candidate))
        aic_with_variate.sort(reverse=True) 
        best_new_score, best_candidate = aic_with_variate.pop()
        if current_score > best_new_score: 
            selected.remove(best_candidate)
            current_score = best_new_score
        else:
            break

    formula = "{}~{}".format(target, "+".join(selected))
    model = smf.ols(formula=formula, data=data).fit()

    return model
