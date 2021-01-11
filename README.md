# Optimisateur-de-portefeuille
Ce programme optimise un portefeuille d'actions en utilisant la notion de frontière efficiente (couple risque/rendement), une notion couverte par la théorie moderne du portefeuille. Dans ce projet, l'optimisation passe par un processus de séléction de la meilleure répartition des actifs possible, parmi toutes les possibilités, selon un objectif. L'objectif dans ce programme est de maximiser le rendement attendu et minimiser les coûts, c'est à dire le risque.

Pour atteindre cet objectif, le programme cherchera donc la répartition du portfeuille de l'utilisateur qui a comme effet de maximiser le ratio Sharpe. Le ratio sharpe est un ratio qui permet d'évaluer le couple rendement/risque. Plus précisément, le ratio Sharpe exprime le rendement excédentaire des actifs (différence entre rendement des actifs et un rendement sans risque, le rendement du marché obligataire est un exemple de rendement sans risque) et divise ce rendement excédentaire par la variance du portefeuille, c'est à dire sa volatilité, qui peut se traduire par le risque. 

Ce programme est interactif. Il demande à l'utilisateur combien de titres composent son portefeuille, et la date d'entrée en bourse la plus récente parmis tous les titres. Le programme calcule par la suite la répartition optimale qui maximise le ratio Sharpe. Une fois cette répartition trouvée (en % pour chacun des actifs), le programme demande combien d'argent l'utilisateur désire investir. Le programme retourne ensuite le nombre d'actions à acheter pour chaque titre afin de respecter la répartition optimale trouvée plus tôt et les liquidités restantes seront aussi affichées. 
