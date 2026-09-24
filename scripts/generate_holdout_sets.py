import os
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

def generate_holdouts():
    os.makedirs("eval", exist_ok=True)
    
    categories = {
        "conversational": [
            "Bună dimineața, mă bucur foarte mult să te revăd astăzi.",
            "Ce mai faci în ultima vreme, totul este în regulă acasă?",
            "Mulțumesc frumos pentru ajutorul acordat, a însemnat enorm pentru mine.",
            "Ne vedem diseară la cafenea sau preferi să rămânem acasă?",
            "Îmi pare rău că am întârziat, traficul a fost absolut imposibil.",
            "Să ai o zi minunată și plină de realizări frumoase alături de cei dragi.",
            "Nu te îngrijora deloc, vom găsi o rezolvare pentru orice problemă.",
            "Putem discuta acest subiect mai pe larg când ai câteva minute libere.",
            "Mă gândeam să pregătesc ceva bun de mâncare pentru masa de seară.",
            "A fost o plăcere sinceră să colaborăm la acest proiect deosebit.",
            "Vremea de afară este superbă, numai bună pentru o plimbare lungă.",
            "Ai reușit să vorbești cu ei despre planurile noastre de vacanță?",
            "Cred că cel mai bine ar fi să luăm o pauză scurtă de zece minute.",
            "Îți urez mult succes la examenul important de săptămâna viitoare.",
            "Dacă ai nevoie de orice altceva, nu ezita să îmi spui imediat.",
            "Am ascultat cu mare atenție tot ce mi-ai povestit mai devreme.",
            "Vom stabili toate detaliile tehnice la întâlnirea de mâine dimineață.",
            "Este remarcabil felul în care ai gestionat această situație delicată.",
            "Mă bucur nespus de mult că am reușit să ne sincronizăm astăzi.",
            "Toate cele bune și ne auzim în curând cu noutăți îmbucurătoare."
        ],
        "a_a_i_heavy": [
            "Păsările călătoare cântau încântător în lumina blândă a dimineții răcoroase.",
            "Bătrânul învățător căuta în cărțile îngălbenite amintiri de demult uitate.",
            "Frământările lăuntrice îl făceau să rătăcească fără țintă prin pădurea deasă.",
            "Cântărețul își încânta ascultătorii cu balade străvechi din bătrâni păstrate.",
            "În adâncul fântânii apele rămâneau reci și neatinse de arșița verii.",
            "Căldura pământului uscat făcea ca aerul să tremure în depărtare.",
            "Învățătura adâncă dobândită în tinerețe îi călăuzea pașii spre înțelepciune.",
            "Mândria strămoșească se oglindea în faptele bărbătești ale oștenilor credincioși.",
            "Vântul bătea aspru prin crengile înghețate ale bătrânilor stejari seculari.",
            "Răsăritul blând lumina cărările înguste ce duceau către mănăstirea din munte.",
            "Încăpățânarea tânărului îl împiedica să asculte sfaturile înțelepte primite.",
            "Căutătorii de comori scormoneau pământul în speranța unei descoperiri mărețe.",
            "Înserarea cădea lin peste cătunul liniștit așezat la poalele dealului.",
            "Bucătăria străbunicii mirosea îmbietor a pâine caldă și plăcinte cu mere.",
            "Învățăceii ascultau tăcuți povestirile pline de tâlc ale bătrânului meșter.",
            "Lăstarii tineri își făceau drum cu îndârjire prin scoarța uscată a copacului.",
            "Împăcarea sufletească vine din acceptarea sinceră a propriilor greșeli din trecut.",
            "Fântânarul săpa cu răbdare până când dădea de izvorul curat și limpede.",
            "Înălțimile munților rămâneau învăluite într-o ceață deasă și nepătrunsă.",
            "Răbdarea și cumpătarea sunt virtuți căutate cu sârguință de oamenii drepți."
        ],
        "s_t_heavy": [
            "Șoaptele liniștite ale serii se pierdeau treptat printre frunzele veștede.",
            "Șoferul conducea cu atenție sporită pe șoseaua îngustă și șerpuitoare.",
            "Școala noastră își deschide porțile pentru toți tinerii dornici de învățătură.",
            "Șarpele se strecura tăcut printre pietrele ascuțite din marginea pădurii.",
            "Știrea neașteptată a stârnit o furtună de reacții în rândul comunității.",
            "Șoimul plutea maiestuos pe cerul senin, scrutând ținutul de la înălțime.",
            "Șansa oferită de destin trebuie prețuită și valorificată cu maximă seriozitate.",
            "Ședința importantă s-a încheiat cu o serie de hotărâri decisive pentru viitor.",
            "Șirul nesfârșit de călători străbătea pasul montan acoperit de zăpadă proaspătă.",
            "Ștefan cel Mare a apărat cu vitejie hotarele țării împotriva invadatorilor.",
            "Țăranul își lucra pământul roditor din zori și până în amurgul liniștit.",
            "Țesătura fină de mătase strălucea în lumina caldă a lumânărilor aprinse.",
            "Ținta stabilită pentru acest trimestru a fost atinsă prin eforturi susținute.",
            "Țara noastră se mândrește cu peisaje pitorești și tradiții populare bogate.",
            "Țipătul ascuțit al vulturului a răsunat ecou peste crestele stâncoase ale masivului.",
            "Țărmul stâncos era bătut neîncetat de valurile înspumate ale mării învolburate.",
            "Țesătorul priceput împletea fire multicolore într-un covor tradițional superb.",
            "Ținuta elegantă a participanților reflecta importanța ceremoniei oficiale organizate.",
            "Țânțarii supărători roiau în jurul lacului liniștit pe timpul nopților calde.",
            "Țintele ambițioase necesită perseverență, disciplină și o strategie bine definită."
        ],
        "affricates": [
            "Cercetătorii au descoperit un cerb magnific care pășea timid prin pădurea seculară.",
            "Gingășia florilor de cireș încânta privirile tuturor călătorilor poposiți în sat.",
            "Ceainicul de porțelan fierbea vesel pe soba din bucătăria mică și primitoare.",
            "Ghețarii milenari din depărtare reflectau razele soarelui într-o lumină feerică.",
            "Chitaristul virtuoz a fermecat publicul cu o interpretare plină de pasiune vie.",
            "Ghioceii firavi își făceau apariția curajoasă de sub stratul alb de nea.",
            "Generalul a dat ordin de atac trupelor cantonate la marginea cetății fortificate.",
            "Ghemul de lână colorată se rostogolea jucăuș pe covorul moale din sufragerie.",
            "Chibritul s-a aprins dintr-o scânteie rapidă, luminând colțul întunecat al camerei.",
            "Geamul înghețat al cabanei era pictat cu flori strălucitoare de ger aprig.",
            "Cina a fost servită într-o atmosferă caldă, acompaniată de o muzică lină.",
            "Ciripitul vesel al păsărelelor anunța sosirea mult așteptată a primăverii timpurii.",
            "Gheața de pe lacul din parc începuse să se topească sub razele blânde.",
            "Ghirlandele luminoase împodobeau centrul vechi al orașului în perioada sărbătorilor.",
            "Chipul senin al copilului radia de bucurie când a deschis cadoul mult visat.",
            "Ghidul montan ne-a condus în siguranță pe traseele abrupte și spectaculoase.",
            "Cerneala neagră s-a așternut cu grijă pe paginile albe ale caietului dictando.",
            "Cinematograful era arhiplin la premiera mult lăudată a noului film românesc.",
            "Ghepardul a alergat cu o viteză uluitoare prin savana întinsă sub soare.",
            "Ghilotina a rămas în cărțile de istorie drept un simbol sumbru al vremurilor trecute."
        ],
        "consonant_clusters": [
            "Strălucirea diamantului șlefuit reflecta spectre spectaculoase de lumină albă.",
            "Construcția noului pod metalic a necesitat eforturi uriașe din partea inginerilor.",
            "Splendoarea arhitecturală a catedralei gotice impresiona orice călător pasionat.",
            "Transportul echipamentelor speciale s-a desfășurat în condiții meteorologice vitrege.",
            "Zdruncinătura puternică a trenului a trezit toți pasagerii din vagonul de dormit.",
            "Strădania continuă a sportivilor s-a concretizat prin obținerea medaliei de aur.",
            "Scânteia izvorâtă din nicovală lumina atelierul fierarului priceput din cătun.",
            "Strămoșii noștri au înfruntat cu bărbăție nenumărate primejdii de-a lungul istoriei.",
            "Schimbul rapid de replici dintre protagoniști a captivat audiența din sală.",
            "Proiectul tehnic include specificații complexe pentru structura de rezistență din beton.",
            "Zdrobit de oboseală după marșul lung, soldatul s-a așezat la umbra copacului.",
            "Împrospătarea forțelor de ordine s-a realizat prin aducerea de noi contingente.",
            "Stropii mari de ploaie au început să cadă ritmic pe acoperișul din tablă zincată.",
            "Privirea ageră a vânătorului urmărea fiecare mișcare suspicioasă din desiș.",
            "Aventurierul a străbătut pe jos ținuturi neumblate și creste montane primejdioase.",
            "Strângerea recoltei s-a încheiat înainte de venirea primelor brume de toamnă.",
            "Descrierea detaliată a monumentului istoric a atras atenția multor cercetători pasionați.",
            "Zgârie-norii moderni dominau peisajul urban al metropolei în plină expansiune.",
            "Frământarea aluatului de cozonac necesită răbdare, putere și ingrediente proaspete.",
            "Strictețea regulamentului militar impune o disciplină riguroasă în rândul recruților."
        ],
        "numbers_dates": [
            "În data de douăzeci și patru septembrie două mii douăzeci și șase vom lansa proiectul.",
            "Prețul total al facturii fiscale este de o mie patru sute cincizeci de lei.",
            "Tranzacția bancară în valoare de cincizeci de mii de euro a fost confirmată cu succes.",
            "Clădirea istorică din centrul capitalei a fost construită în anul o mie nouă sute zece.",
            "La recensământul din anul trecut au fost înregistrate peste trei sute de gospodării.",
            "Temperatura exterioară a scăzut dramatic până la minus cincisprezece grade Celsius.",
            "Viteza maximă admisă pe acest tronson de autostradă este de o sută treizeci de kilometri pe oră.",
            "Profitul net raportat la sfârșitul trimestrului al treilea a crescut cu douăzeci și cinci la sută.",
            "Evenimentul comemorativ va începe exact la ora optsprezece și treizeci de minute.",
            "Compania a livrat peste zece mii cinci sute de pachete în cursul lunii trecute.",
            "Abonamentul lunar costă doar patruzeci și nouă de lei și nouăzeci și nouă de bani.",
            "Distanța totală parcursă pe jos de drumeți a fost de treizeci și doi de kilometri.",
            "Primul zbor transatlantic cu pasageri a avut loc în prima jumătate a secolului al douăzecilea.",
            "Populația orașului depășește în prezent trei sute cincizeci de mii de locuitori.",
            "Terenul agricol are o suprafață măsurată de o sută douăzeci și cinci de hectare roditoare.",
            "Rezervația naturală adăpostește peste două sute de specii rare de păsări migratoare.",
            "Fondurile alocate pentru modernizarea spitalului județean însumează șaptezeci de milioane de lei.",
            "Trenul accelerat numărul o mie șase sute douăzeci și unu va sosi la linia a doua.",
            "Au trecut mai bine de cincizeci de ani de la marea inundație din primăvara aceea.",
            "Vârsta medie a membrilor din echipa de cercetare este de treizeci și patru de ani."
        ],
        "names_places": [
            "Mihai Eminescu a scris poeme nemuritoare la mănăstirea Văratec din ținutul Neamțului.",
            "Orașul Cluj-Napoca este recunoscut drept un centru universitar de prestigiu european.",
            "Castelul Peleș din Sinaia reprezintă o capodoperă a arhitecturii regale românești.",
            "Fluviul Dunărea se varsă în Marea Neagră formând o deltă de o frumusețe rară.",
            "Alexandru Ioan Cuza a înfăptuit Unirea Principatelor Române la douăzeci și patru ianuarie.",
            "Municipiul Timișoara a fost primul oraș din Europa continentală cu iluminat public electric.",
            "Cetatea Sighișoara își păstrează farmecul medieval unic în inima Transilvaniei.",
            "Munții Făgăraș adăpostesc vârful Moldoveanu, cel mai înalt pisc din România.",
            "Ion Creangă a evocat copilăria idilică petrecută în satul Humulești din Moldova.",
            "Bucureștiul este capitala istorică și centrul cultural al României moderne.",
            "Orașul Brașov este străjuit de muntele Tâmpa și de zidurile vechii cetăți săsești.",
            "George Enescu a compus Rapsodiile Române inspirat din folclorul autentic strămoșesc.",
            "Palatul Culturii din Iași domină peisajul urban al vechii capitale a Moldovei.",
            "Mănăstirea Voroneț este celebră în întreaga lume pentru nuanța unică de albastru.",
            "Mircea cel Bătrân a domnit cu demnitate peste Țara Românească timp de trei decenii.",
            "Studiourile de film din Buftea au găzduit producții cinematografice românești celebre.",
            "Orașul Sibiu a fost desemnat capitală culturală europeană datorită patrimoniului său.",
            "Cheile Bicazului oferă o priveliște impresionantă călătorilor care traversează Carpații.",
            "Constantin Brâncuși a sculptat Coloana Infinitului la Târgu Jiu ca omagiu eroilor.",
            "Lacul Roșu este o atracție turistică montană formată în urma unei prăbușiri naturale."
        ],
        "technical_english": [
            "Algoritmul de machine learning optimizează parametrii modelului prin backpropagation eficient.",
            "Am descărcat noul update software direct din cloud pentru serverul principal.",
            "Echipa de development a testat pipeline-ul de continuous integration pe clusterul local.",
            "Driverul plăcii video NVIDIA RTX suportă accelerare hardware prin kernel-uri CUDA.",
            "Am configurat mediul virtual Python folosind package manager-ul ultra-rapid uv.",
            "Framework-ul PyTorch alocă memorie VRAM pentru operațiile de deep learning pe GPU.",
            "Dataset-ul de antrenare conține fișiere audio comprimate în format tokenizat RVQ.",
            "Pentru optimizarea latenței, microserviciul rulează într-un container Docker izolat.",
            "Am salvat greutățile rețelei neuronale ca fișiere safetensors pe disk-ul intern.",
            "Benchmark-ul de performanță a arătat o viteză de procesare de peste șaizeci de FPS.",
            "Interfața grafică a aplicației a fost redesenată folosind componente responsive moderne.",
            "Am integrat un API REST securizat pentru comunicarea dintre backend și frontend.",
            "Procesul de fine-tuning utilizează tehnica LoRA pentru a reduce amprenta de memorie.",
            "Sistemul de operare Windows gestionează memoria video prin arhitectura driverului WDDM.",
            "Pentru evaluarea calității audio am calculat metricile Word Error Rate și Character Error Rate.",
            "Rețeaua transformer include straturi de multi-head attention cu rotary position embedding.",
            "Am rulat un script de monitoring pentru a urmări temperatura și consumul de wattage.",
            "Noul release include bug fix-uri importante și îmbunătățiri de stabilitate software.",
            "Serverul procesează request-uri concurente folosind arhitectura asincronă bazată pe event loop.",
            "Modelul de sinteză vocală produce un waveform audio fidel prin tehnologia neural vocoder."
        ],
        "questions_exclamations": [
            "Oare vom reuși să finalizăm toate experimentele științifice înainte de miezul nopții?",
            "Cât de fascinant este modul în care inteligența artificială poate învăța limba română!",
            "Unde ai găsit aceste informații atât de valoroase și detaliate despre proiect?",
            "Ce realizare extraordinară pentru echipa noastră de tineri cercetători pasionați!",
            "Crezi că este posibil să optimizăm viteza de antrenare fără a pierde din acuratețe?",
            "Incredibil cât de mult a progresat tehnologia în doar câțiva ani de zile!",
            "De ce ai ales această abordare tehnică în locul metodelor convenționale recomandate?",
            "Să nu renunți niciodată la visul tău, indiferent de obstacolele întâlnite pe drum!",
            "Când vom putea asculta primele mostre audio sintetizate de noul nostru model?",
            "Ce minunată este priveliștea munților la apus, văzută de pe creasta aceasta înaltă!",
            "Ai verificat dacă toate fișierele de configurare au fost salvate corect în director?",
            "Bravo, ai demonstrat o perseverență de invidiat pe parcursul întregului maraton de lucru!",
            "Cine ar fi crezut că un calculator de jocuri poate antrena rețele neuronale atât de mari?",
            "Ai văzut cât de limpede și natural sună vocea sintetizată pe eșantioanele de test?",
            "Cum reușește modelul să păstreze pronunția corectă a diacriticelor românești?",
            "Extraordinar, rezultatele obținute depășesc chiar și cele mai optimiste așteptări ale noastre!",
            "Care sunt următorii pași metodologici pe care ar trebui să îi urmăm în cercetare?",
            "Atenție sporită la parametrii de temperatură pentru a preveni supraîncălzirea hardware-ului!",
            "Este oare posibil să extindem această metodă și pentru alte limbi cu resurse puține?",
            "Ce bucurie imensă aduce clipa în care codul tău funcționează exact așa cum ai dorit!"
        ],
        "long_sentences": [
            "Dezvoltarea recentă a modelelor multimodale de mari dimensiuni a deschis noi oportunități remarcabile pentru comunitatea științifică, permițând transpunerea ideilor teoretice complexe în soluții practice inovatoare capabile să înțeleagă și să genereze limbaj natural cu o fidelitate fără precedent.",
            "În ciuda dificultăților tehnice inerente legate de constrângerile de memorie video ale plăcilor grafice moderne, aplicarea metodelor avansate de cuantizare pe patru biți și a tehnicilor de adaptare cu rang scăzut a demonstrat că cercetarea de vârf în domeniul inteligenței artificiale poate fi realizată și pe calculatoare personale.",
            "Păstrarea patrimoniului lingvistic al limbii române în era digitală reprezintă o datorie morală și științifică esențială, motiv pentru care antrenarea unor sisteme de sinteză vocală capabile să redea nuanțele subtile ale diacriticelor și ale intonației native constituie un obiectiv strategic de mare importanță.",
            "Cercetătorii care au analizat structura internă a rețelelor neuronale artificiale au observat că împărțirea sarcinilor cognitive între un modul de gândire de mare capacitate și un modul de vorbire specializat permite obținerea unor rezultate spectaculoase cu un consum substanțial redus de resurse de calcul.",
            "După săptămâni întregi de investigații riguroase și teste preliminare desfășurate în laboratoarele universitare, echipa de tineri ingineri a reușit să demonstreze experimental că puntea dintre spațiul de coduri acustice discrete și sintetizatorul de undă sonoră funcționează cu o precizie matematică impresionantă.",
            "Diversitatea fonetică a eșantioanelor selectate din înregistrările narative de calitate profesională oferă rețelei neuronale un spectru acustic bogat, asigurând învățarea temeinică a tuturor tranzițiilor dintre consoane ocluzive, vocale deschise și consoane fricative caracteristice vorbirii românești cursive.",
            "Deși entuziasmul descoperirilor rapide este adesea copleșitor pentru tinerii pasionați de tehnologie, respectarea cu strictețe a metodei științifice, documentarea transparentă a erorilor întâlnite și validarea ipotezelor prin porți de decizie obiective rămân singurele căi sigure către progrese autentice și durabile.",
            "În liniștea profundă a nopților târzii, când ventilatoarele plăcii grafice murmurau constant disipând căldura generată de milioanele de operații de calcul tensorial, fiecare scădere a funcției de pierdere reprezenta un pas mic, dar sigur, către visul de a învăța inteligența artificială să vorbească românește.",
            "Arhitectura modulară a sistemului de sinteză vocală a permis decuplarea eficientă a fazei de predicție a codurilor audio primare de faza de rafinare a detaliilor acustice secundare, facilitând antrenarea iterativă și optimizarea separată a fiecărei componente fără a periclita stabilitatea modelului de bază.",
            "Prin compararea minuțioasă a transcrierilor automate obținute prin sistemele independente de recunoaștere a vorbirii cu textele de referință rostite de vorbitorul nativ, cercetătorii au putut cuantifica cu exactitate impactul măririi volumului de date de antrenament asupra clarității și inteligibilității sunetului generat.",
            "Peisajul pitoresc al satelor tradiționale românești, așezate cuminți de-a lungul văilor străjuite de păduri seculare de brad și fag, a constituit dintotdeauna o sursă inepuizabilă de inspirație pentru scriitorii și poeții care au știut să surprindă sufletul profund și ospitalitatea caldă a oamenilor acestor locuri.",
            "Procesul de cuantizare a parametrilor neurali pe tipuri de date specializate cu precizie redusă nu doar că diminuează semnificativ cerințele de stocare pe magistrala de memorie, dar deschide și calea către rularea unor algoritmi de ultimă generație pe dispozitive compacte destinate publicului larg.",
            "Odată cu validarea definitivă a compatibilității dintre reprezentarea pe șaisprezece canale de cuantizare reziduală și vocoderul neuronal dedicat, comunitatea de cercetare a primit o confirmare empirică clară că modelele moderne de limbaj pot fi extinse rapid către noi spații lingvistice fără costuri prohibitive.",
            "Munca minuțioasă de verificare a integrității fiecărui tensor salvat în punctele de control periodic garantează că nicio fluctuație neașteptată de tensiune sau eroare temporară de alocare a memoriei nu poate distruge orele îndelungate de efort depuse pentru instruirea ponderilor sinaptice ale rețelei.",
            "Atunci când tânărul elev a văzut pentru prima dată cum propriul calculator asamblează în timp real undele sonore ale unei limbi materne pe care modelul original nu o cunoștea, a înțeles că știința calculatoarelor nu este doar o colecție de formule aride, ci o veritabilă artă a creației digitale.",
            "Colaborarea armonioasă dintre mentorul artificial și discipolul dornic de cunoaștere a transformat camera de lucru a unui pasionat de jocuri video într-un veritabil laborator de cercetare experimentală, unde fiecare test eșuat era întâmpinat cu luciditate și fiecare reușită era consemnată cu rigoare.",
            "Capacitatea extraordinară a sistemelor autonome de a monitoriza în permanență starea componentelor hardware, temperatura circuitelor integrate și consumul energetic asigură desfășurarea neîntreruptă a experimentelor complexe chiar și în absența supravegherii umane directe pentru intervale lungi de timp.",
            "Evaluarea atentă a mostrelor audio de către vorbitori nativi experimentați a confirmat faptul că redarea fluentă a intonației interogative și a accentului melodic al frazelor constituie cheia fundamentală pentru obținerea unei percepții de naturalețe deplină în comunicarea om-mașină.",
            "Încheierea cu succes a ciclurilor succesive de instruire și validare marchează nu un punct final, ci începutul unei etape noi și captivante, în care modelul optimizat va putea fi integrat în asistenți conversaționali utili pentru educație, accesibilitate și asistență medicală comunitară.",
            "Fiecare linie de cod scrisă cu atenție, fiecare fișier de configurare verificat cu grijă și fiecare decizie bazată exclusiv pe dovezi măsurabile contribuie la consolidarea unei culturi inginerești sănătoase, în care adevărul științific primează întotdeauna asupra supozițiilor comode sau a concluziilor grăbite."
        ]
    }
    
    # 1. ro_holdout_200.jsonl
    all_200 = []
    quick_40 = []
    
    total_idx = 1
    for cat_name, sentences in categories.items():
        for i, s in enumerate(sentences):
            rec = {
                "id": f"ro_eval_{total_idx:03d}",
                "category": cat_name,
                "text": s
            }
            all_200.append(rec)
            # Pick 4 from each category for quick 40 (indices 0, 5, 10, 15)
            if i in [0, 5, 10, 15]:
                quick_40.append(rec)
            total_idx += 1
            
    with open("eval/ro_holdout_200.jsonl", "w", encoding="utf-8") as f:
        for r in all_200:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open("eval/ro_holdout_quick_40.jsonl", "w", encoding="utf-8") as f:
        for r in quick_40:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Created eval/ro_holdout_200.jsonl ({len(all_200)} sentences across {len(categories)} categories)")
    print(f"Created eval/ro_holdout_quick_40.jsonl ({len(quick_40)} sentences, exactly 4 per category)")
    
    # Also verify data manifests in data/manifests/
    os.makedirs("data/manifests", exist_ok=True)
    manifest_sources = [
        ("data/manifests/ro_30.jsonl", "reports/mimi_codes.jsonl", 30),
        ("data/manifests/ro_150.jsonl", "reports/mimi_codes_ro150.jsonl", 150),
        ("data/manifests/ro_500.jsonl", "reports/mimi_codes_ro500.jsonl", 500),
        ("data/manifests/ro_1000.jsonl", "reports/mimi_codes_ro1000.jsonl", 1000)
    ]
    for target_m, src_file, limit in manifest_sources:
        if os.path.exists(src_file):
            with open(src_file, "r", encoding="utf-8") as f_in, open(target_m, "w", encoding="utf-8") as f_out:
                count = 0
                for line in f_in:
                    if line.strip():
                        d = json.loads(line)
                        entry = {
                            "sample_id": d["sample_id"],
                            "transcript": d.get("transcript", ""),
                            "duration": d.get("duration", 0),
                            "wav_path": d.get("wav_path", "")
                        }
                        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        count += 1
                        if count >= limit:
                            break
            print(f"Created manifest {target_m} ({count} items)")

if __name__ == "__main__":
    generate_holdouts()
