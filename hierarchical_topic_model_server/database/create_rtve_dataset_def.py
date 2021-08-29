# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             READ_DATA                                  ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import pandas as pd
import numpy as np
import re

import configparser

# Gensim
import gensim
from nltk.tokenize import word_tokenize
# NLTK
from base_dm_sql import BaseDMsql
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.phrases import Phrases
import pickle

##############################################################################
#                              CONFIG FILE                                   #
##############################################################################
file = 'config_database.ini'
config = configparser.ConfigParser()
config.read(file)

dbSERVER = config['credentials']['dbSERVER']
dbUSER = config['credentials']['dbUSER']
dbPASS = config['credentials']['dbPASS']
dbTABLENAME = config['production']['dbTABLENAME']
dbNAME = config['credentials']['dbNAME']

num_txt_files = config['production']['num_txt_files']
num_texts = 20

patterns = [(r'real madrid', 'real_madrid'),
            (r'estados unidos','eeuu'),
            (r'estadounidense', 'eeuu'),
            (r'carmen calvo', 'carmen_calvo'),
            (r'pedro sanchez', 'predo_sánchez'),
            (r'pedro sánchez', 'predo_sánchez'),
            (r'iraní', 'irán'),
            (r'iraníes', 'irán'),
            (r'casa blanca', 'casa_blanca'),
            (r'última', 'último'),
            (r'donald trump', 'donald_trump'),
            (r'reino unido', 'reino_unido'),
            (r'trabajadores', 'trabajador'),
            (r'plan europeo', 'plan_europeo'),
            (r'reino unido', 'reino_unido'),
            (r'guardia civil', 'guardia_civil'),
            (r'actores', 'actor'),
            (r'actrices', 'actor'),
            (r'actriz', 'actor'),
            (r'san javier', 'san javier'),
            (r'conjunto blanco', 'conjunto_blanco'),
            (r'jugadores', 'jugador'),
            (r'días', 'día'),
            (r'países', 'país'),
            (r'española', 'españa'),
            (r'josé luis rodríguez zapatero', 'zapatero'),
            (r'josé maría aznar', 'aznar'),
            (r'puntos', 'punto'),
            (r'sergio rama', 'sergio_rama'),
            (r'christian prieto', 'christian_prieto'),
            (r'matt damon', 'matt_damon'),
            (r'fernando velázquez', 'fernando_velázquez'),
            (r'atlético madrid', 'atlético_madrid'),
            (r'atlético', 'atlético_madrid'),
            (r'toni roldán', 'toni_roldán'),
            (r'university college london', 'university_college_london'),
            (r'golfo méxico', 'golfo_méxico'),
            (r'javier fesser', 'javier_fesser'),
            (r'fernando romay', 'fernando_romay'),
            (r'felipe vi', 'felipe_vi'),
            (r'british airway', 'british_airway'),
            (r'pablo iglesias', 'pablo_iglesias'),
            (r'parlamento europeo', 'parlamento_europeo'),
            (r'isabel ii', 'isabel_ii'),
            (r'juan antonio belloch', 'juan_antonio_belloch'),
            (r'eduardo muñoz', 'eduardo_muñoz'),
            (r'marc gasol', 'marc_gasol'),
            (r'pau gasol', 'pau_gasol'),
            (r'leo suárez', 'leo_suárez'),
            (r'dani rodriguez', 'dani_rodriguez'),
            (r'carlos belmonte', 'carlos_belmonte'),    
            (r'ximo puig', 'ximo_puig'),
            (r'jordi sánchez', 'jordi_sánchez'),
            (r'jordi turull', 'jordi_turull'),
            (r'josep rull', 'josep_rull'),
            (r'josé manuel ballester', 'josé_manuel_ballester'),
            (r'josé manuel villegas', 'josé_manuel_villegas'),
            (r'josé luis gonzález sanchís', 'josé_luis_gonzález_sanchís'),
            (r'enrique llopis', 'enrique_llopis'),
            (r'francisco josé amado', 'francisco_josé_amado'),
            (r'david gonzález', 'david_gonzález'),
            (r'josé antonio', 'josé_antonio'),
            (r'josé muñoz', 'josé_muñoz'),
            (r'josé ignacio díaz', 'josé_ignacio_díaz'),
            (r'jesus ángel garcía bragado', 'jesus_ángel_garcía_bragado'),
            (r'eloy josé mestre', 'eloy_josé_mestre'),
            (r'juan josé imbroda montado', 'juan_josé_imbroda_montado'),
            (r'josé luis fernández', 'josé_luis_fernández'),
            (r'josé antonio monago', 'josé_antonio_monago'),
            (r'josé luis díaz', 'josé_luis_díaz'),
            (r'josé miguel mulet', 'josé_miguel_mulet'),
            (r'josé miguel viñas', 'josé_miguel_viñas'),
            (r'josé corbacho', 'josé_corbacho'),
            (r'juan carlos ortega corbacho', 'juan_carlos_ortega_corbacho'),
            (r'josé sacristán', 'josé_sacristán'),
            (r'josé manuel carbonell', 'josé_manuel_carbonell'),
            (r'maría josé inicio igualado', 'maría_josé_inicio_igualado'),
            (r'josé luis ábalos', 'josé_luis_ábalos'),
            (r'josé mayo', 'josé_mayo'),
            (r'josé antonio labordeta', 'josé_antonio_labordeta'),
            (r'josé luis garci', 'josé_luis_garci'),
            (r'josé ramón pardo', 'josé_ramón_pardo'),
            (r'josé manuel sebastián', 'josé_manuel_sebastián'),
            (r'josé luis ingleses', 'josé_luis_ingleses'),
            (r'josé guardiola', 'josé guardiola'),
            (r'héctor rodríguez', 'héctor_rodríguez'),
            (r'josé manuel villegas', 'josé_manuel_villegas'),
            (r'josé carlos endina', 'josé_carlos_endina'),
            (r'josé vicente mozos', 'josé_vicente_mozos'),
            (r'francisco josé garcía', 'francisco_josé_garcía'),
            (r'josé manuel barreiro', 'josé_manuel _barreiro'),
            (r'josé luis sanz sevilla  ', 'josé_luis_sanz_sevilla'),
            (r'juan josé imbroda', 'juan_josé_imbroda'),
            (r'josé luis díaz', 'josé_luis_díaz'),
            (r'josé lozano', 'josé_lozano'),
            (r'josé manuel calderón', 'josé_manuel_calderón'),
            (r'josé mercé', 'josé_mercé'),
            (r'pepón nieto', 'pepón_nieto'),
            (r'hugo silva', 'hugo_silva'),
            (r'emma suárez', 'emma_suárez'),
            (r'roberto leal', 'roberto_leal'),
            (r'josé antonio ortiz ', 'josé_antonio_ortiz '),
            (r'josé sans ', 'josé_sans'),
            (r'josé manuel calderón ', 'josé_manuel_calderón'),
            (r'josé luis rodríguez jiménez', 'josé_luis_rodríguez_jiménez '),
            (r'josé guirao', 'josé_guirao'),
            (r'josé luis ábalos', 'josé_luis_ábalos'),
            (r'josé manuel villegas', 'josé_manuel_villegas'),
            (r'josé ignacio torreblanca', 'josé_ignacio_torreblanca'),
            (r'josé ricardo alfredo prada', 'josé_ricardo_alfredo_prada'),
            (r'simón antunez', 'simón_antunez'),
            (r'gustavo david carlos', 'gustavo_david_carlos'),
            (r'francisco josé alcaraz', 'francisco_josé_alcaraz'),
            (r'josé luis lópez lacalle', 'josé_luis_lópez_lacalle'),
            (r'josé ángel hevia', 'josé_ángel_hevia'),
            (r'josé guirao', 'josé_guirao'),
            (r'josé miguel', 'josé_miguel'),
            (r'josé luis alcobendas', 'josé_luis_alcobendas'),
            (r'david luque', 'david_luque'),
            (r'ernesto aria', 'ernesto_aria'),
            (r'josé manuel urtai', 'josé_manuel_urtai'),
            (r'josé gabriel vera', 'josé_gabriel_vera'),
            (r'josé manuel', 'josé_manuel'),
            (r'josé miguel viñas', 'josé_miguel_viñas'),
            (r'joséluis garci', 'josé_luis_garci'),
            (r'josé ramón pardo', 'josé_ramón_pardo'),
            (r'josé antonio berrocal', 'josé_antonio_berrocal'),
            (r'josé manuel calderón', 'josé_manuel_calderón'),
            (r'josé félix', 'josé_félix'),
            (r'maría josé sánchez', 'maría_josé_sánchez'),
            (r'josé francisco cobo', 'josé_francisco_cobo'),
            (r'josé manuel ramos', 'josé_manuel_ramos'),
            (r'juan josé padilla', 'juan_josé_padilla'),
            (r'ángel peralta', 'ángel_peralta'),
            (r'josé barea', 'josé_barea'),
            (r'josé luis barba ', 'josé_luis_barba'),
            (r'josé barea', 'josé_barea'),
            (r'josé luis', 'josé_luis'),
            (r'josé luis gaya', 'josé_luis_gaya'),
            (r'josé zorrilla', 'josé_zorrilla'), 
            (r'josé cura', 'josé_cura'),
            (r'josé carreras', 'josé_carreras'),
            (r'josé díaz', 'josé_díaz'),
            (r'josé manuel casado', 'josé_manuel_casado'),
            (r'josé luis barba', 'josé_luis_barba'),
            (r'maría josé segarra', 'maría_josé_segarra'),
            (r'josé manuel maza', 'josé_manuel_maza'),
            (r'josé maría calleja', ' josé_maría_calleja'),
            (r'josé antonio maldonado', 'josé_antonio_maldonado'),
            (r'josé sacristán bárbara', 'josé_sacristán_bárbara'),
            (r'lennie alberto', 'lennie_alberto'),
            (r'gonzalo castro pere', 'gonzalo_castro_pere'),
            (r'macarena sanz', 'macarena_sanz'),
            (r'josé ricardo prada', 'josé_ricardo_prada'),
            (r'josé maría fernández', 'josé_maría_fernández'),
            (r'josé miguel castillo', 'josé_miguel_castillo'),
            (r'manuel rosa rubio', 'manuel_rosa_rubio'),
            (r'josé maría riego', 'josé_maría_riego'),
            (r'josé antonio montero', 'josé_antonio_montero'),
            (r'josé manuel carbonell', 'josé_manuel_carbonell'),
            (r'francisco josé salazar', 'francisco_josé_salazar'),
            (r'josé babel', 'josé_babel'),
            (r'josé antonio griñán', 'josé_antonio_griñán'),
            (r'josé manuel villarejo', 'josé_manuel_villarejo'),
            (r'josé luis ábalos', 'josé_luis_ábalos'),
            (r'josé maría vázquez honrubia', 'josé_maría_vázquez_honrubia'),
            (r'jjosé manuel zelaya', 'josé_manuel_zelaya'),
            (r'josé mujica vásquez', 'josé_mujica_vásquez'),
            (r'josé mikel rico', 'josé_mikel_rico'),
            (r'dani garcía sanz', 'dani_garcía_sanz'),
            (r'iñigo martínez yuri', 'iñigo_martínez_yuri'),
            (r'josé alex', 'josé_alex'),
            (r'josé coronado', 'josé_coronado'),
            (r'josé miguel viñas', 'josé_miguel_viñas'),
            (r'josto maffeo', 'josto_maffeo'),
            (r'josé ramón pardo', 'josé_ramón_pardo'),
            (r'josé manuel maza', 'josé_manuel_maza'),
            (r'josé garcía orozco', 'josé_garcía_orozco'),
            (r'mauricio paniagua flores', 'mauricio_paniagua_flores'),
            (r'josé ángel garcía sáenz', 'josé_ángel_garcía_sáenz'),
            (r'josé manuel urtain', 'josé_manuel_urtain'),
            (r'maría josé segarra', 'maría_josé_segarra'),
            (r'josé maraña', 'josé_maraña'),
            (r'francisco josé gómez romero', 'francisco_josé_gómez_romero'),
            (r'israel gómez romero', 'israel_gómez_romero'),
            (r'josé sánchez rubio', 'josé_sánchez_rubio'),
            (r'josé couso', 'josé_couso'),
            (r'juan josé ballesta', 'juan_josé_ballesta'),
            (r'raul arévalo andrés', 'raul_arévalo_andrés'),
            (r'maría rosso', 'maría_rosso'),
            (r'pablo manolo', 'pablo_manolo'),
            (r'alfonso luis callejo benítez', 'alfonso_luis_callejo_benítez'),
            (r'alexandra jiménez', 'alexandra_jiménez'),
            (r'antonio torre', 'antonio_torre'),
            (r'josé antonio guisasola', 'josé_antonio_guisasola'),
            (r'josé luis ayllón', 'josé_luis_ayllón'),
            (r'josé calvo', 'josé_calvo'),
            (r'josé maría lópez', 'josé_maría_lópez'),
            (r'josé luis cuerda', 'josé_luis_cuerda'),
            (r'antonio giménez rico', 'antonio_giménez_rico'),
            (r'osé antonio álvarez', 'josé_antonio_álvarez'),
            (r'josé francisco molina', 'josé_francisco_molina'),
            (r'maría josé rienda', 'maría_josé_rienda'),
            (r'josé luis arrieta nielsen', 'josé_luis_arrieta_nielsen'),
            (r'josé manuel fuente', 'josé_manuel_fuente'),
            (r'bernardo ruiz', 'bernardo_ruiz'),
            (r'miguel poblet', 'miguel_poblet'),
            (r'josé luis santos', 'josé_luis_santos'),
            (r'enrique cima', 'enrique_cima'),
            (r'josé antonio gonzález linares', 'josé_antonio_gonzález_linares'),
            (r'josé vicente arnaiz', 'josé_vicente_arnaiz'),
            (r'juan josé padilla', 'juan_josé_padilla'),
            (r'josé luis carabias', 'josé_luis_carabias'),
            (r'soraya sáenz', 'soraya_sáenz'),
            (r'enrique crespo', 'enrique_crespo'),
            (r'josé ramón', 'josé_ramón'),
            (r'josé sanchís sinisterra', 'josé_sanchís_sinisterra'),
            (r'josé miguel fernández', 'josé_miguel_fernández'),
            (r'josé luis bayo', 'josé_luis_bayo'),
            (r'josé escolar', 'josé_escolar'),
            (r'josé luis ballester', 'jose_luis_ballester'),
            (r'felipe vi', 'falipe_vi'),
            (r'francisco josé carvajal', 'francisco_josé_carvajal'),
            (r'maría carmen jiménez', 'maría_carmen_jiménez'),
            (r'josé perez', 'josé_pérez'),
            (r'josé maría calviño', 'josé_maría_calviño'),
            (r'josé antonio martínez soler', 'josé_antonio_martínez_soler'),
            (r'leonor garcía álvarez', 'leonor_garcía_álvarez'),
            (r'sandra sutherland', 'sandra_sutherland'),
            (r'jose maría fraguas', 'jose_maría_fraguas'),
            (r'pablo carlos mugica', 'pablo_carlos_mugica'),
            (r'julián garcía josé', 'julián_garcía_josé'),
            (r'pablo maldonado', 'pablo maldonado'),
            (r'josé vicente garcia', 'josé_vicente_garcia'),
            (r'josé manuel olivares', 'josé_manuel_olivares'),
            (r'josé maría martín carpena', 'josé_maría_martín_carpena'),
            (r'josé ángel manuel vega', 'josé_ángel_manuel_vega'),
            (r'estefanía santos', 'estefanía_santos'),
            (r'josé antonio gracia', 'josé_antonio_gracia'),
            (r'josé miguel viñas', 'josé_miguel_viñas'),
            (r'maría josé ordóñez', 'maría_josé_ordóñez'),
            (r'josé ignacio wert', 'josé_ignacio_wert'),
            (r'josé ortiz', 'josé_ortiz'),
            (r'centro san josé guadalajara', 'centro_san_josé_guadalajara'),
            (r'josé christian sánchez', 'josé_christian_sánchez'),
            (r'josé guardiola', 'josé_guardiola'),
            (r'susana jiménez', 'susana_jiménez'),
            (r'josé maría gracias', 'josé_maría_gracias'),
            (r'josé maría gonzález sinde', 'josé_maría_gonzález_sinde'),
            (r'fernando trueba', 'fernando_trueba'),
            (r'antonio fernando rey', 'antonio_fernando_rey'),
            (r'gerardo herrero', 'gerardo_herrero'),
            (r'josé luis borau', 'josé_luis_borau'),
            (r'aitana marisa paredes', 'aitana_marisa_paredes'),
            (r'pablo echenique', 'pablo_echenique'),
            (r'mercedes ángeles', 'mercedes_ángeles'),
            (r'eduardo campoy', 'eduardo_campoy'),
            (r'álex iglesia', 'álex_iglesia'),
            (r'enrique gonzález macho', 'enrique_gonzález_macho'),
            (r'antonio resines', 'antonio_resines'),
            (r'josé antonio josu ternera', 'josé_antonio_josu_ternera'),
            (r'josé manuel franco', 'josé_manuel_franco'),
            (r'maría josé parejo', 'maría_josé_parejo'),
            (r'josé maría roldán', 'josé_maría_roldán'),
            (r'josé luis reputado', 'josé_wluis_reputado'),
            (r'josé ignacio rioja', 'josé_ignacio_rioja'),
            (r'juan jesus viva', 'juan_jesus_viva'),
            (r'josé antonio monago', 'josé_antonio_monago'),
            (r'josé antonio', 'josé_antonio_sánchez'),
            (r'josé enrique fernández moya', 'josé_enrique_fernández_moya'),
            (r'josé ramón lete', 'josé_ramón_lete'),
            (r'josé mata', 'josé_mata'),
            (r'josé luis gayá martín', 'josé_luis_gayá_martín'),
            (r'josé maría pou', 'josé_maría_pou'),
            (r'moby dick', 'moby_dick'),
            (r'amaya iriarte', 'amaya_iriarte'),
            (r'josé luis vega', 'josé_luis_vega'),
            (r'josé romera castillo', 'josé_romera_castillo'),
            (r'josé nicolás romera castillo', 'josé_romera_castillo'),
            (r'josé luis sainz heredia', 'josé_luis_sainz_heredia'),
            (r'josé enrique ', 'josé_enrique '),
            (r'josé carlos', 'josé_carlos'),
            (r'juan josé garcía calvo', 'juan_josé_garcía_calvo'),
            (r'julio diego', 'julio_diego'),
            (r'josé ricardo prada', 'josé_ricardo_prada'),
            (r'juan pablo gonzález', 'juan_pablo_gonzález'),
            (r'maría josé rodríguez', 'maría_josé_rodríguez'),
            (r'josé luis uribarri', 'josé_luis_uribarri'),
            (r'josé ignacio goirigolzarri', 'josé_ignacio_goirigolzarri'),
            (r'maría josé gálvez cardona', 'maría_josé_gálvez_cardona'),
            (r'josé manuel rodríguez carrasco', 'josé_manuel_rodríguez_carrasco'),
            (r'josé miguel viñas', 'josé_miguel_viñas'),
            (r'carlos santos', 'carlos_santos'),
            (r'josé ramón laura', 'josé_ramón_laura'),
            (r'josé mota', 'josé_mota'),
            (r'josé ricardo prada', 'josé_ricardo_prada'),
            (r'ana rivas', 'ana_rivas'),
            (r'josé antonio antón', 'josé_antonio_antón'),
            (r'juan antonio valentín', 'juan_antonio_valentín'),
            (r'ramón pradera reyes', 'ramón_pradera_reyes'),
            (r'patricia pérez', 'patricia_pérez'),
            (r'josé maría romero tejada', 'josé_maría_romero_tejada'),
            (r'josé ribera', 'josé_ribera'),
            (r'josé enrique serrano', 'josé_enrique_serrano'),
            (r'josé antonio bermudez castro', 'josé_antonio_bermudez_castro'), #1900 
            (r'juan antonio madrid pérez', 'juan_antonio_madrid_pérez'),
            (r'juan carlos fernández lópez', 'juan_carlos_fernández_lópez'),
            (r'manuel marcos delgado', 'manuel_marcos_delgado'),
            (r'juan carlos rivero', 'juan_carlos_rivero'),
            (r'albert font', 'albert_font'),
            (r'paco caro', 'paco_caro'),
            (r'juanma romero', 'juanma_romero'),
            (r'luis oliván', 'luis_oliván'),
            (r'ana roldán', 'ana_roldán'),
            (r'juan lainez', 'juan_lainez'),
            (r'juan carlos', 'juan_carlos'),
            (r'juan antonio bayona', 'juan_antonio_bayona'),
            (r'juan pablo riesgo', 'juan_pablo_riesgo'),
            (r'juana rivas', 'juana_rivas'),
            (r'juan fernández sánchez', 'juan_fernández_sánchez'),
            (r'leonardo torres quevedo', 'leonardo_torres_quevedo'),
            (r'juan zamora', 'juan_zamora'),
            (r'san juan', 'san_juan'),
            (r'juan arriaga', 'juan_arriaga'),
            (r'juan march', 'juan_march'),
            (r'juan lainez', 'juan_lainez'),
            (r'juan ignacio zoido', 'juan_ignacio_zoido'),
            (r'san climent', 'san_climent'),
            (r'juan carlos ortega', 'juan_carlos_ortega'),
            (r'paco roca', 'paco_roca'),
            (r'don juan carlos', 'don_juan_carlos'),
            (r'juan martín potro', 'juan_martín_potro'),
            (r'juan carlos navarro', 'juan_carlos_navarro'),
            (r'juan carlos orenga', 'juan_carlos_orenga'),
            (r'juan eslava galán', 'juan_Eslava_galán'),
            (r'juan vidal martín', 'juan_vidal_martín'),
            (r'juanjo lobato', 'juanjo_lobato'),
            (r'juan carlos varela', 'juan_carlos_varela'),
            (r'juan blanco', 'juan_blanco'),
            (r'ruth díaz', 'ruth_díaz'),
            (r'juan manuel', 'juan_manuel'),
            (r'juan gonzález', 'juan_gonzález'),
            (r'juan manuel hidalgo', 'juan_manuel_hidalgo'),
            (r'juan padrón', 'juan_padrón'),
            (r'juan camus machado', 'juan_camus_machado'),
            (r'manu tenorio', 'manu_tenorio'),
            (r'juan francisco rosa', 'juan_francisco_rosa'),
            (r'juan josé padilla', 'juan_josé_padilla'),
            (r'juan josé', 'juan_josé'),
            (r'josé_luis rodríguez_zapatero','josé_luis_rodríguez_zapatero'),
            (r'mireia belmonte', 'mireia_belmonte')]

columns = ['id','breadCrumbRef','contentType','htmlShortUrl', 'htmlUrl', 'ids', 'image', 'imageSEO', 
           'language', 'comentariosRef', 'encuestaDestacadaRef', 'encuestasRelacionadasRef', 'encuestasTotemRef',
           'estadisticasRef', 'galeriasRelacionadasRef', 'galeriasTotemRef', 'tagsRef', 'longTitle',
           'mainCategory', 'mainCategoryLang', 'mainCategoryRef', 'newsRelatedRef', 'numVisits', 
           'otherTopicsRef', 'popHistoric','popularity', 'code', 'description' , 'publicationDate',
           'publicationDateTimestamp', 'refreshSeconds', 'relatedByLangRef', 'firma', 'numComentarios', 
           'numCompartidas', 'summary', 'text','textLemmatized', 'tickerNews', 'tickerSports', 'title', 'uri']

def replace(text,patterns):
    for(raw,rep) in patterns:
        regex = re.compile(raw)
        text = regex.sub(rep,text)
    return text

##########################################################################################################################################

corpus_pickle = 'C:\\Users\\lcalv\\OneDrive\\Documentos\\MASTER\\TFM_teleco\\rtve_corpus.pickle'
infile = open(corpus_pickle,'rb')
rtve_data  = pickle.load(infile)   

# LOAD DATA
# # DATABASE CONNECTION
# DB = BaseDMsql(db_name=dbNAME, db_connector='mysql', path2db=None,
#                 db_server=dbSERVER, db_user=dbUSER, db_password=dbPASS)

# # INSERT DATA INTO THE DATABASE
# DB.insertInTable(dbTABLENAME, columns, rtve_data, chunksize=1000, verbose=True)

# #SHOW DATABASE SAVED INFORMATION
# [cols, n_rows] = DB.getTableInfo(dbTABLENAME)
# print(cols)
# print(n_rows)


DB = BaseDMsql(db_name=dbNAME, db_connector='mysql', path2db=None,
               db_server=dbSERVER, db_user=dbUSER, db_password=dbPASS)

# READ DATA
lemmatized_texts = []
n_tokens = []
for df in DB.readDBchunks(dbTABLENAME, 'id', chunksize=50000, selectOptions='textLemmatized', limit=int(num_txt_files)*num_texts, filterOptions=None, verbose=True):
    news_df  = pd.DataFrame(df, columns = ['textLemmatized'])
    # Get lemmatized texts
    for i in np.arange(0,len(news_df),1):
        text = news_df.values[i].tolist()
        text = [replace(word,patterns) for word in text]
        text_str = ', '.join(text)
        text_toks = word_tokenize(text_str)
        #tok_ok = [token for token in text_toks if token not in stopwords]      
        n_tokens.append(len(text_toks))
        lemmatized_texts.append(text_toks)

print("Lematized texts")
print(len(lemmatized_texts))
x = list(np.arange(0,300,20))
bins = list(np.arange(0,300,5)-0.5)
plt.hist(n_tokens,bins=bins,color='#2a9d8f')
plt.xticks(x,x)
plt.axvline(x=80,color='r',linestyle='--',linewidth=3)
plt.xlabel('Number of tokens')
plt.ylabel('Number of documents')
plt.show() # Based on this figure, we remove documents with less tokens than min_tokens = 80

#We do not add to the corpus the documents with less than `min_tokens` = 80 tokens
corpus = []
min_tokens = 80
for i in np.arange(0,len(lemmatized_texts),1):
  if n_tokens[i] >= min_tokens:
    corpus.append(lemmatized_texts[i])
    
print("Final corpus")
print(len(corpus))


phrase_model = Phrases(corpus, min_count=2, threshold=20)
corpus = [el for el in phrase_model[corpus]] 
corpus_def = corpus
print("Len of the final corpus:",len(corpus_def))

# Create dictionary
D_def = gensim.corpora.Dictionary(corpus_def) # Created from all tokens
n_tokens_init = len(D_def)

print('Len of the init dictionary', n_tokens_init)
print('First terms in the dictionary:')
for n in range(10):
  print(str(n), ':', D_def[n])
  
no_below = 5 # Minimum number of documents in which a term must appear to keep this term in the dictionary
no_above = 0.75 # Maximum proportion of documents in which a term can appear to be kept in the dictionary
D_def.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
n_tokens_def = len(D_def)
print('Len of the filtered dictionary', n_tokens_def)
print('First terms in the dictionary:')
for n in range(10):
  print(str(n), ':', D_def[n])
  
dictionary_filtered_list = []
[dictionary_filtered_list.append(D_def.get(i)) for i in np.arange(0,len(D_def),1)] # convertir a set
  
corpus_bow_def = [D_def.doc2bow(doc) for doc in corpus_def]


# SORTED TOKEN FREQUENCIES (I):
# Create a "flat" corpus with all tuples in a single list
corpus_bow_flat = [item for sublist in corpus_bow_def for item in sublist]
n_tokens = len(D_def)

# Initialize a numpy array that we will use to count tokens.
# token_count[n] should store the number of ocurrences of the n-th token, D[n]
token_count = np.zeros(n_tokens)

# Count the number of occurrences of each token.
for x in corpus_bow_flat:
  # Update the proper element in token_count
  token_count[x[0]] += x[1]

# Sort by decreasing number of occurences
idf_sorted = np.argsort(- token_count)
tf_sorted = token_count[idf_sorted]

# SORTED TOKEN FREQUENCIES (II):
plt.rcdefaults()

# Example data
n_art = 5
n_bins = 25
hot_tokens = [D_def[i] for i in idf_sorted[n_bins-1::-1]]
y_pos = np.arange(len(hot_tokens))
z = tf_sorted[n_bins-1::-1]

plt.figure()
plt.barh(y_pos, z, align='center', alpha=0.4, color='#2a9d8f', edgecolor='#2a9d8f')
plt.yticks(y_pos, hot_tokens)
plt.xlabel('Total number of occurrences')
plt.title('Token distribution')
plt.show()

# SORTED TOKEN FREQUENCIES (III):

# Example data
plt.figure()
plt.semilogy(tf_sorted, '#2a9d8f')
plt.ylabel('Total number of occurrences')
plt.xlabel('Token rank')
plt.title('Token occurrences')
plt.show()

indexes = np.arange(0, len(corpus_def))
text_to_save = 'C:\\mallet\\data_news_txt_all_merged.txt'
#text_to_save = 'C:\\mallet\\data_news_txt_all_merged_test.txt'
with open(text_to_save, 'w', encoding='utf-8') as fout:
    i = 0
    for el in corpus_def:
        #final_text = []
        #[final_text.append(text_tok) for text_tok in el if (text_tok in dictionary_filtered_list)]
        fout.write(str(indexes[i]) + ' 0 ' + ' '.join(el) + '\n')
        i += 1