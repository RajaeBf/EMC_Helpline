# Chatbot EMC-Helpline — Assistant conversationnel à base de règles (Jalon 2)

Prototype **sans IA générative, sans RAG, sans LangChain et sans base
vectorielle**, conforme au document de cadrage du stage (15 juillet –
31 août 2026). Chaque mot de chaque réponse envoyée à l'utilisateur provient
exclusivement de `data/knowledge_base.json` : rien n'est jamais généré, ni
reformulé, à la volée — le chatbot ne fait qu'assembler des morceaux de texte
déjà écrits et validés (réponse officielle + éventuels préfixe de politesse,
phrase d'empathie, question de suivi).

## Améliorations apportées à cette version

1. **Tolérance renforcée aux fautes de frappe / abréviations, y compris pour
   l'urgence.** La détection du protocole d'urgence (`detecter_urgence` dans
   `engine/chatbot.py`) tolère désormais les fautes de frappe sur les
   mots-clés de sécurité ("sucide", "mourire"...), alors qu'avant elle ne
   fonctionnait que sur une correspondance exacte. L'abréviation SMS très
   courante "jpp" (j'en peux plus) a aussi été ajoutée aux mots-clés
   d'urgence dans `data/knowledge_base.json`.
2. **Ton plus doux.** Le message par défaut et le message de clarification
   ont été reformulés pour être plus chaleureux et moins "je n'ai pas
   compris" ; une phrase d'empathie a été ajoutée à l'intent
   `Collecte_Preuves`.
3. **Compréhension des réponses "oui"/"non" à une question de suivi.**
   Quand une réponse du chatbot se termine par une question ("Souhaitez-vous
   en savoir plus ?"), le tour suivant est maintenant compris comme une
   réponse à CETTE question précise (voir `_detecter_confirmation` et le
   nouveau paramètre `attente_confirmation` de `ChatbotEMC.repondre()`), avec
   une vraie réponse détaillée en cas de "oui" (champ `suivi_oui` dans
   `knowledge_base.json`) et un message doux en cas de "non" (`suivi_non`).
   Les questions de suivi qui n'apportaient rien de plus que la réponse déjà
   donnée ont été retirées (`Signalement_Abus_Sexuel_Enfant`,
   `Enfant_En_Danger_Assistance`, `Porter_Plainte`), pour ne poser une
   question que quand elle a réellement du sens.
   Un ajustement mineur a aussi été fait pour qu'un message qui mélange une
   salutation et une vraie question (ex : "bjr, on me harcèle sur Instagram,
   koi faire ?") ne se limite plus à un simple "bonjour".

## Architecture

```
emc_chatbot/
├── data/
│   └── knowledge_base.json      # Questions-réponses, mots-clés, intents sociaux,
│                                 # empathie et questions de suivi (Jalon 1)
├── engine/
│   ├── pretraitement.py         # Correction orthographique + lemmatisation
│   └── chatbot.py               # Pipeline complet : score de confiance, clarification,
│                                 # intents sociaux, mémoire de conversation, repli
│                                 # sémantique optionnel, logs de débogage (Jalon 2)
├── tests/
│   └── test_chatbot.py          # Messages types, mémoire de conversation, structure
│                                 # des réponses (empathie/question de suivi)
├── app.py                       # Interface de discussion Streamlit + mémoire de
│                                 # session (st.session_state) (Jalon 3)
├── generer_knowledge_base.py    # Script utilitaire de (re)génération du JSON
├── requirements.txt
└── README.md
```

## Le pipeline NLP, étape par étape

```
Message utilisateur
      │
      ▼
1. Prétraitement            (minuscules, accents et ponctuation retirés)
      │
      ▼
2. Correction orthographique légère     <- engine/pretraitement.py
      │
      ▼
3. Suppression des stopwords + racinisation (lemmatisation légère)
      │              (branche séparée, utilisée pour un score de "sens")
      ▼
4. Recherche par mots-clés (exacte + tolérance aux fautes)
      │  + score de recouvrement lexical ("lemmes")
      ▼  → combinés en un score de confiance unique (0 à 1)
5. Recherche par similarité sémantique (SentenceTransformer, optionnelle)
      │  → utilisée seulement si le niveau 4 ne suffit pas
      ▼
6. Décision selon le niveau de confiance :
      - HAUTE (≥ 0.60)   → réponse officielle de l'intent
      - MOYENNE (≥ 0.30)  → proposition de 2-3 intentions proches
      - BASSE (< 0.30)    → message par défaut (reformuler)
```

Le protocole d'urgence reste vérifié **avant tout le reste**, sans exception.

### 1-2. Prétraitement + correction orthographique (`engine/pretraitement.py`)

Le texte est mis en minuscules, les accents et la ponctuation sont retirés,
puis chaque mot est corrigé :

- **Dictionnaire de corrections courantes** (`CORRECTIONS_COURANTES`) : abréviations
  SMS ("ui"→"qui", "pk"→"pourquoi", "svp"→"s'il vous plaît") et fautes très
  fréquentes sur le vocabulaire du domaine ("facbook"→"facebook",
  "instagrame"→"instagram", "harclement"→"harcelement"...).
- **Correction floue basée sur le vocabulaire** : un vocabulaire de référence
  est construit automatiquement à partir de tous les mots-clés et questions-types
  de `knowledge_base.json`. Un mot inconnu (≥ 4 lettres) est comparé (via
  `difflib`, bibliothèque standard) aux mots du vocabulaire commençant par la
  même lettre ; s'il ressemble à plus de 82 % à l'un d'eux, il est corrigé.
  *(Exiger la même première lettre est une règle classique de correction
  orthographique : une faute de frappe change très rarement la première
  lettre d'un mot — cela évite par exemple que "entrer" soit corrigé à tort
  en "entre".)*

### 3. Suppression des stopwords + racinisation

Une liste de ~90 mots-outils français (articles, pronoms, prépositions,
auxiliaires être/avoir...) est retirée du texte. Chaque mot restant est
ramené à une racine approximative par `raciniser()` : suppression de
suffixes courants (conjugaisons, pluriels, formes en *-ement*/*-age*) +
une petite table de verbes irréguliers fréquents (pouvoir, vouloir, devoir,
savoir, valoir). Exemples :

```
gère / gérer / gérerai         -> "ger"
dormir                          -> "dorm"
vaux / vaut / valoir            -> "val"   (verbe irrégulier)
harcèlement / harceler          -> "harcel"
```

Ce n'est **pas** un lemmatiseur linguistiquement parfait, juste des règles
suffisantes pour regrouper les variantes les plus fréquentes rencontrées
dans ce projet — sans aucune dépendance externe (uniquement `difflib` et des
expressions régulières de la bibliothèque standard).

Cette branche "lemmes" tourne **en parallèle** de la recherche par mots-clés
classique (qui, elle, continue de fonctionner sur le texte complet, pour ne
jamais casser les phrases exactes déjà présentes dans la base) : elle sert
de second signal, capable de reconnaître une reformulation qui ne partage
aucune phrase exacte avec la base (ex: la négation orale "je n'arrive plus"
devient souvent "j'arrive plus" à l'oral — "ne" étant de toute façon un
stopword, les deux formulations donnent le même ensemble de lemmes).

### 4. Score de confiance combiné

Pour chaque intent, deux signaux sont calculés puis combinés :

- **score mots-clés** : comme avant (correspondance exacte ou approximative
  des mots-clés), normalisé entre 0 et 1.
- **score lemmes** : proportion des mots porteurs de sens du message qui se
  retrouvent (une fois racinisés) parmi les mots-clés/questions de l'intent.

Ils sont combinés par une formule de type "probabilités indépendantes" :

```
confiance = score_mots_cles + (1 − score_mots_cles) × 0.75 × score_lemmes
```

Concrètement : un bon score de mots-clés (≥ 1 mot-clé exact) suffit déjà à
répondre directement ; un très fort recouvrement de lemmes (formulation
totalement différente, mais évoquant clairement les mêmes idées) peut, à lui
seul, faire franchir le seuil de confiance haute — sans qu'aucune phrase
exacte n'ait été anticipée dans la base.

### 5. Repli sémantique (optionnel, inchangé dans son principe)

Si aucun intent n'atteint la confiance haute, et si le repli sémantique est
disponible *et activé* (voir plus bas), le message est comparé par
similarité de sens (modèle `SentenceTransformer` multilingue) aux
questions-types de la base. **Ce n'est ni du RAG ni un générateur de texte** :
un simple calcul de similarité cosinus entre deux phrases, utilisé pour
choisir laquelle des réponses déjà écrites est la plus proche.

⚠️ **Désactivé par défaut** (même si `sentence-transformers` est installé),
pour éviter tout blocage sur un hébergement aux ressources limitées (voir
section dédiée plus bas). À activer via `EMC_ACTIVER_SEMANTIQUE=1` si besoin.

### 6. Décision selon la confiance — clarification au lieu d'un refus sec

Trois zones de confiance (configurables à la création du `ChatbotEMC`) :

| Zone | Seuil par défaut | Comportement |
|---|---|---|
| **Haute** | ≥ 0.60 | Réponse officielle de l'intent, directement |
| **Moyenne** | ≥ 0.30 | Message proposant les 2-3 intentions les plus proches (avec un exemple de question pour chacune), au lieu de "je n'ai pas compris" |
| **Basse** | < 0.30 | Message par défaut, invitant à reformuler |

Exemple de réponse en zone "moyenne" (message : *"problème avec mon compte"*) :

```
Je ne suis pas totalement certain(e) de bien comprendre votre question. Vouliez-vous plutôt dire :
1. Où signaler du cyberharcèlement sur les réseaux sociaux ?
2. Comment signaler un compte Instagram ?
3. C'est quoi EMC-Helpline ?

N'hésitez pas à reformuler votre question en vous inspirant d'une de ces formulations.
```

*Limite assumée* : le chatbot ne peut pas (encore) traiter "1" comme une
sélection directe de la première proposition — l'utilisateur doit reformuler
en s'inspirant d'un des exemples, ou poser une question de suivi naturelle
(voir la mémoire de conversation ci-dessous, qui couvre le cas le plus
fréquent : poursuivre sur le même sujet après une première réponse).

## Assistant conversationnel : salutations, empathie, mémoire de conversation

Au-delà du moteur de recherche de FAQ, le chatbot adopte maintenant un
comportement plus proche d'un véritable assistant d'accompagnement — tout en
restant 100 % à base de règles : aucune phrase n'est générée, tout provient
de `data/knowledge_base.json`.

### Intents sociaux (gérés séparément du moteur métier)

Quatre tournures de politesse sont détectées indépendamment des questions
métier, via `intents_sociaux` dans la base : **Salutation**, **Remerciement**,
**Au_Revoir**, **Compliment**. Elles ont leur propre détection par mots-clés
(`_detecter_intent_social`), plus simple que le pipeline métier (pas besoin
de lemmes ni de sémantique pour reconnaître "merci" ou "au revoir").

- Si **seule** une tournure sociale est détectée (ex: "Bonjour" tout seul) →
  réponse autonome de l'intent social (champ `"reponse"`).
- Si une tournure sociale **et** une vraie question métier sont détectées
  dans le même message (ex: *"bonjour présentez-vous"*) → la réponse
  officielle est précédée d'un court préfixe de politesse (champ
  `"prefixe"`), au lieu d'ignorer la salutation.

```
>>> "bonjour présentez-vous"

Bonjour 👋, j'espère que vous allez bien.

Espace Maroc Cyberconfiance (EMC) est une initiative du Centre Marocain...
```

### Réponses structurées (empathie + question de suivi)

Certains intents du moteur métier (piratage de compte, revenge porn,
cyberharcèlement, dépôt de plainte, conséquences psychologiques...) possèdent
désormais deux champs optionnels dans `knowledge_base.json` :

- `"empathie"` : une courte phrase professionnelle validant la situation de
  l'utilisateur, affichée avant la réponse officielle.
- `"question_suivi"` : une question naturelle pour poursuivre l'échange,
  affichée après la réponse officielle.

La réponse finale suit toujours le même ordre :

```
[Salutation éventuelle]

[Empathie éventuelle]

[Réponse officielle EMC — jamais modifiée]

[Question de suivi éventuelle]
```

**La réponse officielle elle-même n'est jamais reformulée ni raccourcie** :
`_construire_reponse_intent()` se contente d'assembler des morceaux de texte
déjà écrits et validés, tous issus de la base de connaissances. Les intents
purement informatifs (présentation de l'EMC, définitions, bonnes pratiques...)
n'ont volontairement pas de champ `empathie`/`question_suivi` : en forcer un
partout rendrait le ton artificiel.

### Mémoire de conversation

`ChatbotEMC.repondre(message, intent_precedent=...)` accepte un second
paramètre optionnel : l'identifiant du dernier intent métier reconnu dans la
conversation. Cela permet de comprendre une question de suivi qui, seule,
serait trop vague pour être orientée :

```
Utilisateur : Mon compte Facebook est piraté.
Bot         : [réponse Signalement_Reseaux_Sociaux]

Utilisateur : Et après ?
Bot         : [même réponse, methode="memoire-contexte"]
```

Le déclenchement reste volontairement **prudent et explicite** : la mémoire
n'est utilisée que si (a) un intent précédent est transmis, ET (b) le message
correspond à une expression de continuation générique reconnue
(`EXPRESSIONS_CONTINUATION` dans `engine/chatbot.py` : "et après", "comment
faire", "quelle est la procédure"...). Si le message contient un vrai signal
métier différent, ce nouveau signal est toujours prioritaire sur la mémoire.

Côté interface (`app.py`), la mémoire est stockée dans
`st.session_state.dernier_intent` et mise à jour uniquement quand une réponse
métier confiante a été donnée (pas pour une salutation seule, une
clarification ou le message par défaut).

*Limite assumée* : pas de résolution de références plus complexes ("et pour
lui aussi ?", "la même chose pour Instagram") — seules les formulations de
continuation générique de la liste ci-dessus sont reconnues. Étendre cette
liste, ou ajouter une compréhension plus fine des références, serait une
extension possible du Jalon 3.

## Logs de débogage

Pour comprendre pourquoi une intention a été choisie (ou pas), activez le
mode debug avec la variable d'environnement `EMC_DEBUG` :

```bash
EMC_DEBUG=1 python engine/chatbot.py
```

Chaque message affiche alors : le texte corrigé, les lemmes extraits, et le
classement des 3 intents les plus proches avec leur score de confiance :

```
[EMC-Helpline] Message='facbook pirate' | corrige='facebook pirate' | lemmes=['facebook', 'pirat'] | top3=[('Signalement_Reseaux_Sociaux', 0.68), ('Salutation', 0.0), ('EMC_Presentation', 0.0)]
[EMC-Helpline] -> Réponse directe : intent=Signalement_Reseaux_Sociaux (confiance=0.68, methode=mots-cles)
```

En dehors du mode debug, le chatbot reste silencieux (niveau `WARNING`).

## Performance

Le pipeline est pensé pour rester rapide, même avec la correction
orthographique et la lemmatisation :

- Vocabulaire, mots-clés normalisés et ensembles de lemmes par intent sont
  **précalculés une seule fois** au démarrage (`ChatbotEMC.__init__`), jamais
  recalculés à chaque message.
- Le vocabulaire de correction est **indexé par première lettre** : un mot
  inconnu n'est comparé qu'aux mots du vocabulaire commençant par la même
  lettre, pas à tout le vocabulaire.
- Les tokens du message sont **dédupliqués** avant la recherche par
  mots-clés (évite de recomparer plusieurs fois un mot répété).
- `raciniser()` (la fonction de racinisation) est mise en cache (`lru_cache`).

Résultat mesuré sur un message typique : **~2-3 ms**. Même un message
inhabituellement long (100+ mots, beaucoup de mots inconnus) reste sous les
40 ms — largement suffisant pour un usage conversationnel.

## Comment ça marche : le moteur en résumé

1. Le message est normalisé, corrigé (orthographe), puis analysé par deux
   voies en parallèle : mots-clés (texte corrigé) et lemmes (sens des mots).
2. Le protocole d'urgence est vérifié en priorité absolue.
3. Le score de confiance combiné détermine si on répond directement, si on
   propose une clarification, ou si on affiche le message par défaut.
4. Le repli sémantique (désactivé par défaut) peut compléter le niveau 4
   quand rien n'est assez confiant.

## Dépendance optionnelle (repli sémantique)

Le Niveau 5 nécessite la librairie `sentence-transformers` (voir
`requirements.txt`). **Si elle n'est pas installée** (ou si le modèle ne peut
pas être téléchargé, par exemple hors connexion), le chatbot continue de
fonctionner normalement avec les Niveaux 1 à 4 (mots-clés + lemmes),
sans planter.

⚠️ **Sur un hébergement gratuit aux ressources limitées** (ex : Streamlit
Community Cloud, souvent ~1 Go de RAM), le premier chargement du modèle
(PyTorch + téléchargement ~120 Mo) peut être lent, voire faire planter ou
geler l'application. Pour cette raison, **le Niveau 5 est désactivé par
défaut**, même si la librairie est installée, tant que la variable
d'environnement `EMC_ACTIVER_SEMANTIQUE` n'est pas explicitement mise à `"1"` :

```bash
# En local, pour tester le repli sémantique :
export EMC_ACTIVER_SEMANTIQUE=1        # Linux/Mac
$env:EMC_ACTIVER_SEMANTIQUE = "1"      # Windows PowerShell

streamlit run app.py
```

Sur Streamlit Community Cloud, cette variable se règle dans les paramètres
de l'app (section "Secrets").

**En pratique, le pipeline mots-clés + lemmes (Niveaux 1 à 4) résout déjà,
seul, la quasi-totalité des reformulations naturelles testées** — le repli
sémantique reste une option pour aller encore plus loin, pas un prérequis.

## Lancer le projet

```bash
# Dépendance de base pour l'interface (le repli sémantique est optionnel, voir ci-dessus)
pip install streamlit

# Tester le moteur en ligne de commande
python engine/chatbot.py
# ... ou avec le détail du calcul de confiance :
EMC_DEBUG=1 python engine/chatbot.py

# Lancer la suite de tests avec des messages types
python tests/test_chatbot.py

# Lancer l'interface de discussion
streamlit run app.py

# Régénérer data/knowledge_base.json après une modification du script
python generer_knowledge_base.py
```

## Comment ajouter une nouvelle question (sans toucher au code)

Ouvrir `generer_knowledge_base.py` (ou directement `data/knowledge_base.json`)
et ajouter un nouvel objet dans la liste `"intents"`, sur ce modèle :

```json
{
  "id": "Mon_Nouvel_Intent",
  "categorie": "informatif",
  "questions_types": ["Exemple de question posée par l'utilisateur"],
  "mots_cles": ["mot cle 1", "mot cle 2", "expression complete"],
  "reponse": "La réponse officielle, validée par l'encadrant.",
  "reference": "Section correspondante du document source"
}
```

Grâce au nouveau pipeline (correction orthographique + lemmes), il n'est
**plus nécessaire d'énumérer toutes les fautes d'orthographe ou variantes
grammaticales possibles** : quelques mots-clés bien choisis suffisent
généralement, le reste est couvert automatiquement. Il reste en revanche
utile d'ajouter un mot-clé quand il s'agit d'un **concept réellement absent**
de la base (ex : nous avons dû ajouter quelques mots-clés pour "compte
piraté / accès perdu", un scénario qui n'existait pas du tout auparavant —
voir section suivante).

### Ajouter une phrase d'empathie ou une question de suivi à un intent existant

Ajouter simplement les champs optionnels `"empathie"` et/ou `"question_suivi"`
à l'intent concerné (voir `Signalement_Revenge_Porn_Adulte` par exemple) :

```json
{
  "id": "Mon_Intent_Sensible",
  "categorie": "technique",
  "empathie": "Je suis désolé(e) que vous traversiez cette situation.",
  "question_suivi": "Souhaitez-vous connaître la procédure détaillée ?",
  "...": "... (reste de l'intent inchangé)"
}
```

Ces deux champs sont **facultatifs** : un intent qui ne les possède pas reçoit
simplement sa réponse officielle sans phrase d'ouverture ni question de fin
(c'est volontairement le cas des intents purement informatifs).

### Ajouter ou modifier un intent social

Les tournures de politesse vivent dans `intents_sociaux` (et non `intents`),
sur un modèle proche mais avec un champ `"prefixe"` supplémentaire (utilisé
quand la tournure sociale accompagne une vraie question métier) :

```json
{
  "id": "Mon_Intent_Social",
  "mots_cles": ["mot 1", "mot 2"],
  "prefixe": "Courte phrase utilisée en préfixe d'une réponse métier.",
  "reponse": "Réponse complète utilisée quand SEUL ce social est détecté."
}
```

## Changements apportés à `data/knowledge_base.json`

Le rôle du fichier n'a pas changé (il reste l'unique source des réponses, des
mots-clés, et maintenant aussi des phrases d'empathie/questions de suivi).
Changements apportés lors de cette évolution :

- **Nouvelle section `intents_sociaux`** : Salutation, Remerciement,
  Au_Revoir, Compliment — déplacés hors de `intents` (ils n'étaient traités
  auparavant que comme un intent métier "salutation" parmi d'autres).
- **Champs `"empathie"` et `"question_suivi"`** ajoutés à 12 intents
  sensibles (piratage/harcèlement de compte, abus sur enfant, contenu intime
  mineur, revenge porn, enfant en danger, aspects juridiques, conséquences
  psychologiques, signaux d'alerte parents, définition du cyberharcèlement,
  stratégie victime). Volontairement absents des intents purement
  informatifs (présentation de l'EMC, définitions générales, bonnes
  pratiques...) pour ne pas rendre le ton artificiel.
- **`Signalement_Reseaux_Sociaux`** : mots-clés pour le scénario "compte
  piraté / accès perdu" (concept qui n'était couvert par aucun intent
  existant).
- **`Definition_Cyberharcelement`** : mots-clés décrivant le comportement
  ("ça dure depuis des semaines") en plus des questions méta.

## Limites actuelles (à documenter dans le rapport de stage)

- La racinisation (`raciniser()`) est une heuristique à base de suffixes, pas
  un lemmatiseur linguistique complet : elle ne gère pas tous les verbes
  irréguliers du français (seuls les plus fréquents/pertinents pour ce
  projet ont été ajoutés à la main).
- La correction orthographique peut, rarement, corriger un mot correct vers
  un mot proche mais différent (faux positif) ; le seuil de ressemblance
  (0.82) et l'exigence de première lettre identique limitent ce risque sans
  l'éliminer totalement.
- La zone de confiance "moyenne" propose une clarification mais ne peut pas
  encore interpréter une réponse du type "1" ou "2" à cette proposition. La
  mémoire de conversation couvre un cas différent et plus simple (reprendre
  le dernier intent métier confiant sur une question de suivi générique) ;
  elle ne résout pas des références plus fines ("et pour Instagram aussi ?").
- Une seule langue (français). L'arabe reste une extension possible (le
  modèle sémantique optionnel est déjà multilingue, ce qui faciliterait
  cette extension).
- Les seuils (`seuil_confiance_haute`, `seuil_confiance_moyenne`,
  `seuil_similarite`, `seuil_fuzzy`) sont des valeurs de départ raisonnables,
  ajustées empiriquement sur les exemples testés ; ils peuvent nécessiter un
  réglage après des tests avec de vrais utilisateurs (Jalon 3).

## Prochaines étapes possibles

- Ajouter la version arabe des réponses (mots-clés + racinisation + questions-types).
- Ajouter une mémoire de conversation légère pour permettre de répondre "1"
  à une proposition de clarification.
- Étendre la table de verbes irréguliers si de nouveaux cas d'usage l'exigent.
- Activer le repli sémantique en production une fois hébergé sur une
  infrastructure avec plus de ressources.
# EMC_Helpline
