import nltk
from nltk import CFG
from nltk.tree import Tree
from nltk.grammar import Nonterminal, Production


class CYK:

    def __init__(self, grammar):
        self.grammar = grammar

    def to_cnf(self, built_in_cnf=False):
        if built_in_cnf:
            self.grammar = CFG.chomsky_normal_form(self=self.grammar, new_token_padding='X', flexible=False)
        else:
            grammar = self.grammar
            new_rule_counter = 1

            # Rule of type A -> B C D...
            for prod in grammar.productions(): # Recursive because new productions are appended
                if prod.is_nonlexical() and len(prod.rhs()) > 2:
                    xhs = Nonterminal('X' + str(new_rule_counter))
                    new_rule_counter += 1
                    new_prod1 = Production(prod.lhs(), [xhs, prod.rhs()[2]])
                    grammar.productions().append(new_prod1)
                    new_prod2 = Production(xhs, [prod.rhs()[0], prod.rhs()[1]])
                    grammar.productions().append(new_prod2)
                    grammar.productions().remove(prod)

            # Rule of type A -> s B (or B s)
            for prod in grammar.productions():
                if prod.is_lexical() and len(prod.rhs()) == 2:
                    if type(prod.rhs()[0]) is str:
                        xhs = Nonterminal('X' + str(new_rule_counter))
                        new_rule_counter += 1
                        new_prod1 = Production(prod.lhs(), [xhs, prod.rhs()[1]])
                        grammar.productions().append(new_prod1)
                        new_prod2 = Production(xhs, [prod.rhs()[0]])
                        grammar.productions().append(new_prod2)
                        grammar.productions().remove(prod)
                    if type(prod.rhs()[1]) is str:
                        xhs = Nonterminal('X' + str(new_rule_counter))
                        new_rule_counter += 1
                        new_prod1 = Production(prod.lhs(), [prod.rhs()[0], xhs])
                        grammar.productions().append(new_prod1)
                        new_prod2 = Production(xhs, [prod.rhs()[1]])
                        grammar.productions().append(new_prod2)
                        grammar.productions().remove(prod)

            # Rule of type A -> B
            for prod in grammar.productions():
                if prod.is_nonlexical() and len(prod.rhs()) == 1:
                    for rule in grammar.productions():
                        if prod.rhs()[0] == rule.lhs():
                            lhs = Nonterminal(prod.lhs())
                            new_prod = Production(lhs, [rule.rhs()[0]])
                            grammar.productions().append(new_prod)
                    grammar.productions().remove(prod)

            self.grammar = grammar

    def parse(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        grammar = self.grammar
        # Initialize table of trees
        number_tokens = len(tokens)
        tree_table = [[[] for i in range(number_tokens + 1)] for j in range(number_tokens + 1)]
        for i in range(number_tokens):
            productions = []
            for prod in grammar.productions():
                if prod.rhs()[0] == tokens[i]: productions.append(prod.lhs())
            if not productions:
                print("ERROR: CFG does not have lexical rule for: '" + tokens[i] + "' in " + '"' + sentence + '"')
                return []
            for j in range(len(productions)):
                tree_table[i][i + 1].append(Tree(productions[j], [tokens[i]]))

        # Main algorithm
        for span in range(1, number_tokens + 1):
            for start_pt in range(number_tokens - span + 1):
                end_pt = start_pt + span
                for split_pt in range(start_pt + 1, end_pt):
                    nonterminals1, nonterminals2 = tree_table[start_pt][split_pt], tree_table[split_pt][end_pt]
                    if nonterminals1 and nonterminals2:  # if cell is not empty
                        for NT1 in nonterminals1:
                            for NT2 in nonterminals2:
                                for prod in grammar.productions():
                                    if (NT1.label(), NT2.label()) == prod.rhs():
                                        if 'X' in str(NT1.label()):
                                            tree_table[start_pt][end_pt].append(Tree(prod.lhs(), [NT1[0], NT1[1], NT2]))
                                        elif 'X' in str(NT2.label()):
                                            tree_table[start_pt][end_pt].append(Tree(prod.lhs(), [NT1, NT2[0], NT2[1]]))
                                        else:
                                            tree_table[start_pt][end_pt].append(Tree(prod.lhs(), [NT1, NT2]))
        trees = tree_table[0][number_tokens]
        for tree in trees:
            print(tree.pformat())
            tree.draw()
        if not trees: print('ERROR: No valid syntax for "' + sentence + '" in given CFG')
        return trees

french_grammar = CFG.fromstring("""
    N-M-sg -> 'chat' | 'livre' | 'poisson'
    N-M-pl -> 'chats' | 'livres' | 'poissons'
    N-F-sg -> 'télévision' | 'semaine' | 'fleur' | 'plante'
    N-F-pl -> 'télévisions' | 'semaines' | 'fleurs' | 'plantes'
    
    PR-1sg -> 'je'
    PR-2sg -> 'tu'
    PR-3sg -> 'il' | 'elle'
    PR-1pl -> 'nous'
    PR-2pl -> 'vous'
    PR-3pl -> 'ils' | 'elles'
    
    V-13sg -> 'mange' | 'regarde' | 'parle'
    V-2sg -> 'manges' | 'regardes' | 'parles'
    V-1pl -> 'mangeons' | 'regardons' | 'parlons'
    V-2pl -> 'mangez' | 'regardez' | 'parlez'
    V-3pl -> 'mangent' | 'regardent' | 'parlent'
    
    NEGne -> 'ne'
    NEGpas -> 'pas'
    
    DT-M -> 'le'
    DT-F -> 'la'
    DT-pl -> 'les'
    DOP -> 'le' | 'la' | 'les'
    
    PN-DT -> 'Canada' | 'France' | 'Finlande'
    PN -> 'Daniel' | 'Montréal' | 'Jupiter'
    
    A-M-sg-posN -> 'noir' | 'amusant' | 'heureux' 
    A-M-pl-posN -> 'noirs' | 'amusants' | 'heureux'
    A-F-sg-posN -> 'noire' | 'amusante' | 'heureuse'
    A-F-pl-posN -> 'noires' | 'amusantes' | 'heureuses'
    A-M-sg-preN -> 'gros' | 'beau' | 'joli'
    A-F-sg-prepostN -> 'dernière' | 'prochaine' | 'propre'
    
    
    S -> NP-13sg VP-13sg | PR-1pl VP-1pl | PR-2sg VP-2sg | PR-2pl VP-2pl | NP-3pl VP-3pl
    
    NP-M -> DT-M A-M-sg-preN N-M-sg | DT-M N-M-sg A-M-sg-posN | DT-M N-M-sg | DT-M PN-DT | PN
    NP-F -> DT-F N-F-sg A-F-sg-posN | DT-F A-F-sg-prepostN N-F-sg | DT-F N-F-sg A-F-sg-prepostN | DT-F N-F-sg
    NP-pl -> DT-pl N-M-pl A-M-pl-posN | DT-pl N-F-pl A-F-pl-posN | DT-pl N-F-pl | DT-pl N-M-pl
    NP -> NP-M | NP-F | NP-pl
    
    NP-13sg -> NP-M | NP-F | PR-1sg | PR-3sg
    NP-3pl -> NP-pl | PR-3pl
    
    PR -> PR-1sg | PR-2sg | PR-3sg | PR-1pl | PR-2pl | PR-3pl
    
    VP-13sg -> V-13sg NP | DOP V-13sg | VPneg-13sg NP
    VP-2sg -> V-2sg NP | DOP V-2sg | VPneg-2sg NP
    VP-1pl -> V-1pl NP | DOP V-1pl | VPneg-1pl NP
    VP-2pl -> V-2pl NP | DOP V-2pl | VPneg-2pl NP
    VP-3pl -> V-3pl NP | DOP V-3pl | VPneg-3pl NP
    
    VPneg-13sg -> NEGne V-13sg NEGpas
    VPneg-2sg -> NEGne V-2sg NEGpas
    VPneg-1pl -> NEGne V-1pl NEGpas
    VPneg-2pl -> NEGne V-2pl NEGpas
    VPneg-3pl -> NEGne V-3pl NEGpas
    """)

cyk_parser = CYK(french_grammar)
cyk_parser.to_cnf(built_in_cnf=True) # Toggle True to use built-in CFG.chomsky_normal_form() function
sentence = "je mange le poisson"

cyk_parser.parse(sentence)