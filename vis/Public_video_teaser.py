import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


annotations = [1, 1, 1, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
predictions = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
softmax = [[0.7786824107170105, 0.2213175743818283], [0.9980833530426025, 0.0019167017890140414], [0.9983888864517212, 0.001611122745089233], [0.9433906674385071, 0.0566093772649765], [0.9544076919555664, 0.0455922931432724], [0.17501436173915863, 0.8249856233596802], [0.999566376209259, 0.0004336067067924887], [0.9957594275474548, 0.0042406124994158745], [0.9974205493927002, 0.0025795127730816603], [0.974658191204071, 0.025341864675283432], [0.5418565273284912, 0.4581434428691864], [0.9999121427536011, 8.789816638454795e-05], [0.3415844440460205, 0.6584155559539795], [0.9458672404289246, 0.05413278192281723], [0.9937830567359924, 0.006217007525265217], [0.9990371465682983, 0.000962862279266119], [0.9992610812187195, 0.0007388632511720061], [0.7725435495376587, 0.22745640575885773], [0.9995182752609253, 0.00048179025179706514], [0.018545027822256088, 0.9814550280570984], [0.9998661279678345, 0.0001338990987278521], [0.8980812430381775, 0.1019187644124031], [0.9997918009757996, 0.00020823303202632815], [0.999466598033905, 0.0005334126180969179], [0.963532567024231, 0.03646748512983322], [0.9830030798912048, 0.01699690707027912], [0.8573204278945923, 0.14267955720424652], [0.20275098085403442, 0.7972490191459656], [0.9856840372085571, 0.014315906912088394], [0.9998743534088135, 0.00012557725131046027], [0.5237494707107544, 0.4762504994869232], [0.8799458146095276, 0.12005414068698883], [0.03839738294482231, 0.9616026282310486], [0.8331261277198792, 0.16687381267547607], [0.999987006187439, 1.2952860743098427e-05], [0.9999783039093018, 2.168173887184821e-05], [0.9999092817306519, 9.071378008229658e-05], [0.6124648451805115, 0.3875351846218109], [0.999977707862854, 2.230481550213881e-05], [7.7564109233208e-06, 0.9999922513961792], [0.9950531125068665, 0.00494691077619791], [0.04702742025256157, 0.9529725909233093], [0.7117887139320374, 0.28821131587028503], [0.03765521198511124, 0.962344765663147], [0.9995575547218323, 0.0004423742357175797], [0.9999265670776367, 7.343172183027491e-05], [2.4462219698762055e-06, 0.9999974966049194], [0.9989173412322998, 0.0010826286161318421], [0.9997139573097229, 0.00028603451210074127], [0.9513497352600098, 0.04865031689405441], [0.9986013770103455, 0.0013986671110615134], [0.9998830556869507, 0.00011687207734212279], [0.7740852236747742, 0.22591480612754822], [0.035484421998262405, 0.9645155668258667], [0.8458982706069946, 0.15410171449184418], [0.9968198537826538, 0.0031800989527255297], [0.8329312205314636, 0.167068749666214], [0.9998019337654114, 0.0001980765227926895], [0.9996311664581299, 0.0003688668366521597], [0.8121105432510376, 0.1878894418478012], [0.3982928693294525, 0.6017071604728699], [0.9993855953216553, 0.0006143782520666718], [0.9999579191207886, 4.2133273382205516e-05], [0.9993371367454529, 0.0006627861293964088], [0.014194494113326073, 0.9858054518699646], [0.033143848180770874, 0.9668561220169067], [0.9002096652984619, 0.09979038685560226], [0.9997110962867737, 0.0002889197494369], [0.4583245813846588, 0.5416754484176636], [0.9997298121452332, 0.0002701150078792125], [0.9992272853851318, 0.0007726640906184912], [0.8929298520088196, 0.10707011818885803], [0.866158664226532, 0.13384129106998444], [0.9999386072158813, 6.140318146208301e-05], [0.9990384578704834, 0.0009615164599381387], [0.9993995428085327, 0.0006004253518767655], [0.09043776988983154, 0.9095622301101685], [0.6561425924301147, 0.34385740756988525], [0.9992902278900146, 0.0007097495254129171], [0.9998307228088379, 0.00016921249334700406], [0.9373757839202881, 0.06262422353029251], [0.6989607214927673, 0.30103930830955505], [0.9998231530189514, 0.00017683886107988656], [0.9987282156944275, 0.0012717758072540164], [0.9995693564414978, 0.00043063831981271505], [0.9844776391983032, 0.015522393397986889], [0.9927738904953003, 0.0072261313907802105], [0.9984937906265259, 0.0015061880694702268], [0.9999134540557861, 8.651416283100843e-05], [0.9999415874481201, 5.838027209392749e-05], [0.9646705389022827, 0.035329513251781464], [0.12866030633449554, 0.8713396191596985], [0.9997654557228088, 0.00023452592722605914], [0.9934579730033875, 0.006541987881064415], [0.9999780654907227, 2.1916486730333418e-05], [0.9961490631103516, 0.0038509161677211523], [0.016186576336622238, 0.9838134050369263], [0.9190544486045837, 0.08094550669193268], [0.9027840495109558, 0.09721598029136658], [0.0031760777346789837, 0.9968239068984985], [0.9185618162155151, 0.08143821358680725], [0.9999929666519165, 7.092245141393505e-06], [0.9944332242012024, 0.005566806998103857], [0.5395793318748474, 0.4604206681251526], [0.9087724685668945, 0.09122749418020248], [0.5684010982513428, 0.43159884214401245], [8.959787010098808e-07, 0.9999990463256836], [0.999919056892395, 8.091831114143133e-05], [0.9994094371795654, 0.0005905834259465337], [0.05339807644486427, 0.9466018676757812], [0.9973211884498596, 0.002678836928680539], [0.9999027252197266, 9.726257849251851e-05], [0.033443305641412735, 0.9665566682815552], [0.7408592104911804, 0.2591407597064972], [0.9996762275695801, 0.0003237882046960294], [0.9999814033508301, 1.8641931092133746e-05], [0.9999430179595947, 5.696112202713266e-05], [0.999518871307373, 0.0004811848921235651], [0.8877406120300293, 0.1122593805193901], [0.9277023077011108, 0.07229769974946976], [0.9999966621398926, 3.350027100168518e-06], [0.9751949906349182, 0.024805011227726936], [0.8113527894020081, 0.18864719569683075], [0.9992871880531311, 0.0007128664292395115], [0.9999812841415405, 1.869860352599062e-05], [0.9999853372573853, 1.4705188732477836e-05], [0.9976143836975098, 0.002385692670941353], [0.9945397973060608, 0.005460167769342661], [0.9990726709365845, 0.0009273213800042868], [0.7879698276519775, 0.21203015744686127], [0.9999706745147705, 2.9367529350565746e-05], [0.9999157190322876, 8.426279237028211e-05], [0.999685525894165, 0.0003145195369143039], [0.9987320303916931, 0.001268017222173512], [0.9994596838951111, 0.0005403207615017891], [0.3197636604309082, 0.6802363395690918], [0.016883596777915955, 0.9831164479255676], [0.9999629259109497, 3.703636321006343e-05], [0.9999943971633911, 5.606649210676551e-06], [0.025368696078658104, 0.9746313095092773], [0.9965665340423584, 0.003433383535593748], [0.9991130232810974, 0.0008869703742675483], [0.9473899006843567, 0.052610088139772415], [0.8375768065452576, 0.16242317855358124], [0.9915288686752319, 0.008471077308058739], [0.9998511075973511, 0.00014891002501826733], [0.9987528324127197, 0.0012471789959818125], [0.9999333620071411, 6.663577369181439e-05], [0.9999457597732544, 5.4206546337809414e-05], [0.999977707862854, 2.2312688088277355e-05], [0.9993984699249268, 0.000601485779043287], [0.1525728851556778, 0.8474271297454834], [0.9994500279426575, 0.0005499345134012401], [0.02787170372903347, 0.9721283316612244], [0.9999862909317017, 1.3679919902642723e-05], [0.892034649848938, 0.10796532779932022], [0.9999849796295166, 1.4989867850090377e-05], [0.9998536109924316, 0.0001463571679778397], [0.9989995360374451, 0.001000462332740426], [0.9995086193084717, 0.0004913702141493559], [0.9999645948410034, 3.5395249142311513e-05], [0.8586676120758057, 0.14133243262767792], [0.9998107552528381, 0.00018926300981547683], [0.9871633052825928, 0.012836698442697525], [0.9998651742935181, 0.00013482733629643917], [0.03313774615526199, 0.9668622612953186], [0.10166467726230621, 0.898335337638855], [0.9597166776657104, 0.040283363312482834], [0.9151185154914856, 0.0848815068602562], [0.9997947812080383, 0.0002051883639069274], [0.9956621527671814, 0.0043378486298024654], [0.9977594614028931, 0.0022405132185667753], [0.9999134540557861, 8.653297845739871e-05], [0.9996869564056396, 0.00031308468896895647], [0.9997580647468567, 0.0002419646189082414], [0.9997921586036682, 0.00020787635003216565], [0.9999274015426636, 7.261725841090083e-05], [0.20707111060619354, 0.7929288744926453], [0.04191860929131508, 0.9580813646316528], [0.0606326200067997, 0.9393673539161682], [0.6547321677207947, 0.34526780247688293], [0.9997220635414124, 0.00027794166817329824], [0.9994751811027527, 0.0005248220986686647], [0.9946746826171875, 0.005325353238731623], [0.9999581575393677, 4.189152241451666e-05], [0.9879814982414246, 0.012018534354865551], [0.9616149067878723, 0.03838507458567619], [0.541968584060669, 0.45803138613700867], [0.9169378876686096, 0.08306210488080978], [0.8967076539993286, 0.103292316198349], [0.9986932873725891, 0.001306672696955502], [0.3482409715652466, 0.6517590284347534], [0.9997342228889465, 0.0002657841541804373], [0.021870305761694908, 0.9781297445297241], [0.9935739636421204, 0.0064259921200573444], [0.9987416863441467, 0.0012582839699462056], [0.014219671487808228, 0.9857802987098694], [0.9714263081550598, 0.02857370674610138], [0.9785295724868774, 0.021470485255122185], [0.9998263716697693, 0.00017369413399137557], [0.9998812675476074, 0.00011875671771122143], [0.9993680119514465, 0.0006320278625935316], [0.8935826420783997, 0.10641735792160034], [0.03117220848798752, 0.9688277840614319], [0.9995254278182983, 0.0004746133054140955], [0.9975415468215942, 0.002458451548591256], [0.025508970022201538, 0.9744910597801208], [0.5775734186172485, 0.4224265515804291], [0.2960008978843689, 0.7039991021156311], [0.9993118047714233, 0.0006882367306388915], [0.9942753911018372, 0.005724560469388962], [0.9877117276191711, 0.012288243509829044], [0.9984027743339539, 0.001597263035364449], [0.9994630217552185, 0.0005369886057451367], [1.0805379133671522e-06, 0.999998927116394], [0.997963547706604, 0.0020364366937428713], [0.9995793700218201, 0.00042059042607434094], [0.999963641166687, 3.639194983406924e-05], [0.9980747699737549, 0.0019252222264185548], [0.025567758828401566, 0.9744321703910828], [0.9984081387519836, 0.001591879059560597], [0.6857457756996155, 0.3142542243003845], [0.9997989535331726, 0.00020103060523979366], [0.999995231628418, 4.815886768483324e-06], [0.9999808073043823, 1.9151513697579503e-05], [0.9999886751174927, 1.1356664799677674e-05], [0.9988946318626404, 0.001105440896935761], [0.9748502969741821, 0.02514970488846302], [0.9897356033325195, 0.010264430195093155], [0.03652553632855415, 0.963474452495575], [0.012183799408376217, 0.9878161549568176], [0.12472516298294067, 0.8752748370170593], [0.9999219179153442, 7.80481132096611e-05], [0.8688502311706543, 0.1311497837305069], [0.9997946619987488, 0.00020529990433715284], [0.9998892545700073, 0.0001107331772800535], [0.8886125087738037, 0.11138748377561569], [0.9988805651664734, 0.0011194683611392975], [0.8753010034561157, 0.12469907104969025], [0.9980940222740173, 0.0019060260383412242], [0.9997851252555847, 0.00021485923207364976], [0.9998624324798584, 0.00013750822108704597], [1.2850489383708918e-06, 0.9999986886978149], [0.9256449937820435, 0.07435499876737595], [0.9997382760047913, 0.0002617716963868588], [0.9746782183647156, 0.02532179094851017], [0.015914559364318848, 0.9840854406356812], [0.9994612336158752, 0.0005387861165218055], [0.7704112529754639, 0.22958879172801971], [0.9996278285980225, 0.00037211121525615454], [0.9984226226806641, 0.001577412593178451], [0.9972532391548157, 0.002746752230450511], [0.999964714050293, 3.5253058740636334e-05], [0.8860390186309814, 0.11396093666553497], [0.9983605742454529, 0.0016393921105191112], [0.9948204755783081, 0.005179546307772398], [0.999813973903656, 0.00018610009283293039], [0.014023227617144585, 0.9859768152236938], [0.12538619339466095, 0.8746137619018555], [0.9780358672142029, 0.02196415886282921], [0.7121808528900146, 0.28781914710998535], [0.9994768500328064, 0.0005231631803326309], [0.9996046423912048, 0.0003953429404646158], [0.5186089873313904, 0.4813910126686096], [0.9810365438461304, 0.018963415175676346], [0.998981773853302, 0.0010182171827182174], [0.9738196134567261, 0.026180338114500046], [0.9960917830467224, 0.003908239305019379], [0.9994151592254639, 0.0005848990986123681], [0.9999774694442749, 2.2513517251354642e-05], [0.999823272228241, 0.00017675459093879908], [0.19817425310611725, 0.801825761795044], [0.9999740123748779, 2.5931747586582787e-05], [0.9997990727424622, 0.00020092329941689968], [0.06320640444755554, 0.9367935657501221], [0.401782363653183, 0.5982176661491394], [0.9999722242355347, 2.7799093004432507e-05], [0.9999418258666992, 5.816344128106721e-05], [0.6797890663146973, 0.3202109634876251], [0.999349057674408, 0.0006509278318844736], [0.021757105365395546, 0.9782428741455078], [0.9999231100082397, 7.693299266975373e-05], [0.9996757507324219, 0.0003243149258196354], [0.9962234497070312, 0.0037764860317111015], [0.41386085748672485, 0.5861391425132751], [0.9997263550758362, 0.0002735823218245059], [0.9992043375968933, 0.0007956931367516518], [0.5223815441131592, 0.47761842608451843], [0.9991944432258606, 0.0008055108482949436], [0.9954313039779663, 0.004568664822727442], [0.9998086094856262, 0.00019145815167576075], [0.9842758178710938, 0.015724167227745056], [0.16096211969852448, 0.8390378355979919], [0.9978659749031067, 0.002134032780304551], [0.988243043422699, 0.01175699383020401], [0.9999620914459229, 3.796057353611104e-05], [0.9610596299171448, 0.038940418511629105], [0.9075000882148743, 0.09249994158744812], [0.8033109903335571, 0.19668900966644287], [0.9992256164550781, 0.000774433312471956], [0.9988347887992859, 0.0011652124812826514], [0.9643781781196594, 0.03562181442975998], [0.0509493350982666, 0.9490507245063782], [0.9996165037155151, 0.0003835046954918653], [0.9784157872200012, 0.02158423513174057], [0.2382831871509552, 0.7617167830467224], [0.7892807126045227, 0.21071931719779968], [0.9990923404693604, 0.0009076659916900098], [0.06799519807100296, 0.9320048093795776], [0.9994530081748962, 0.000546941242646426], [0.9983018636703491, 0.0016980755608528852], [0.9999545812606812, 4.540765439742245e-05], [0.9915846586227417, 0.008415277116000652], [0.9994658827781677, 0.000534175313077867], [0.9971280694007874, 0.002871986012905836], [0.9996688365936279, 0.0003311769396532327], [0.99836665391922, 0.0016333753010258079], [0.9997819066047668, 0.00021810809266753495], [0.9995914101600647, 0.0004085942928213626], [0.9998345375061035, 0.00016542641969863325], [0.11040711402893066, 0.8895928859710693], [0.06744866818189621, 0.9325513243675232], [0.9999010562896729, 9.89287145785056e-05], [0.998040497303009, 0.001959536923095584], [0.9994159936904907, 0.0005839670775458217], [0.8162825107574463, 0.1837175190448761], [0.9997026324272156, 0.0002973077935166657], [0.9932128190994263, 0.006787160877138376], [0.88978111743927, 0.11021888256072998], [0.98497074842453, 0.01502927578985691], [0.9999189376831055, 8.109598275041208e-05], [0.9514621496200562, 0.04853780195116997], [0.2816508710384369, 0.7183490991592407], [0.787093997001648, 0.21290600299835205], [0.9962144494056702, 0.003785487962886691], [0.9965143799781799, 0.0034856710117310286], [0.9997345805168152, 0.00026539649115875363], [0.9991683959960938, 0.000831593933980912], [0.9993869066238403, 0.0006131294067017734], [0.9514793157577515, 0.04852067306637764], [2.957895503641339e-06, 0.9999970197677612], [0.9999898672103882, 1.017318936646916e-05], [0.9456196427345276, 0.05438036099076271], [0.8528622388839722, 0.14713774621486664], [0.22956737875938416, 0.7704326510429382], [0.999812662601471, 0.0001873994478955865], [0.9996907711029053, 0.00030928789055906236], [0.01523991022258997, 0.9847601056098938], [0.9995643496513367, 0.00043565567466430366], [0.9074013829231262, 0.09259862452745438], [0.9999903440475464, 9.67341384239262e-06], [0.9993724226951599, 0.0006275674095377326], [0.9444599151611328, 0.0555400475859642], [0.9871093034744263, 0.012890659272670746], [0.12753833830356598, 0.8724616765975952], [0.9972321391105652, 0.0027678327169269323], [0.9999873638153076, 1.2679543942795135e-05], [0.8175186514854431, 0.1824812889099121], [0.9999804496765137, 1.954844810825307e-05], [0.9993498921394348, 0.0006501401658169925], [0.71452796459198, 0.2854720652103424], [0.9997262358665466, 0.0002737488248385489], [0.9999949932098389, 5.018648607801879e-06], [0.9998273253440857, 0.00017266436771024019], [0.6129662990570068, 0.38703370094299316], [0.996040940284729, 0.003959106281399727], [0.8733808994293213, 0.12661908566951752], [0.8816094994544983, 0.11839049309492111], [0.46001535654067993, 0.5399846434593201], [0.9995356798171997, 0.0004643702122848481], [0.989048182964325, 0.010951836593449116], [0.7140475511550903, 0.28595247864723206], [0.7667229175567627, 0.2332770973443985], [0.8629992604255676, 0.13700081408023834], [0.959954559803009, 0.04004548117518425], [0.9780024886131287, 0.021997472271323204], [0.07533685117959976, 0.9246631860733032], [0.9838981032371521, 0.016101837158203125], [0.9999974966049194, 2.4661721909069456e-06], [0.6128569841384888, 0.38714301586151123], [0.9991857409477234, 0.0008142204023897648], [0.854113757610321, 0.14588621258735657], [0.7743133902549744, 0.22568656504154205], [0.9999446868896484, 5.527255780179985e-05], [0.9999630451202393, 3.696741259773262e-05], [0.9896530508995056, 0.010346983559429646], [0.9653447866439819, 0.034655191004276276], [0.9993626475334167, 0.0006373928626999259], [0.07543237507343292, 0.9245676398277283], [0.39672839641571045, 0.6032716631889343], [0.9996607303619385, 0.00033930662903003395], [0.9998327493667603, 0.00016727445472497493], [0.9988303780555725, 0.0011695909779518843], [0.8343542814254761, 0.16564573347568512], [0.9993863105773926, 0.0006136478623375297], [0.9982790946960449, 0.001720862346701324], [0.9999736547470093, 2.63903166342061e-05], [0.9845553040504456, 0.015444711782038212], [0.9951469302177429, 0.0048530930653214455], [0.9999604225158691, 3.9583850593771785e-05], [0.9999374151229858, 6.2540166254621e-05], [0.9998815059661865, 0.00011845892004203051], [0.023234877735376358, 0.9767650961875916], [0.9999544620513916, 4.558510772767477e-05], [0.9995471835136414, 0.0004527962300926447], [0.9961899518966675, 0.0038099864032119513], [0.9972231388092041, 0.002776889596134424], [0.9997296929359436, 0.0002703293866943568], [0.836166262626648, 0.16383373737335205], [0.999631404876709, 0.0003685532428789884], [0.999954104423523, 4.587795046973042e-05], [0.9520733952522278, 0.047926586121320724], [0.9999208450317383, 7.92022910900414e-05], [0.06504186242818832, 0.9349581599235535], [0.9927653670310974, 0.0072346399538218975], [0.9999823570251465, 1.7615251636016183e-05], [0.997123658657074, 0.002876299899071455], [0.9989076852798462, 0.0010923409136012197], [0.9852914810180664, 0.014708477072417736], [0.3624790608882904, 0.637520968914032], [0.9999680519104004, 3.1933992431731895e-05], [0.9950941801071167, 0.0049058785662055016], [0.8550090193748474, 0.144990935921669], [0.9824979901313782, 0.01750202104449272], [0.9998496770858765, 0.00015030776557978243], [0.9961118102073669, 0.003888186998665333], [0.9046990871429443, 0.09530085325241089], [0.9993693232536316, 0.000630751543212682], [0.994786262512207, 0.0052137114107608795], [0.9991682767868042, 0.0008317675674334168], [0.9997832179069519, 0.000216850676224567], [0.9966955184936523, 0.003304491750895977], [0.9999983310699463, 1.6306651104969205e-06], [0.6480220556259155, 0.3519779145717621], [0.9851114153862, 0.01488861721009016], [0.008709260262548923, 0.991290807723999], [0.9998760223388672, 0.0001239328703377396], [0.9776695370674133, 0.02233045920729637], [0.9996768236160278, 0.0003231602895539254], [0.9998883008956909, 0.00011174363316968083], [0.24541343748569489, 0.7545865178108215], [0.6342412233352661, 0.3657587468624115], [0.0228547565639019, 0.9771451950073242], [0.998145580291748, 0.0018544216873124242], [0.8074237704277039, 0.19257624447345734], [0.9941416382789612, 0.005858369171619415], [0.9995492100715637, 0.0004507523844949901], [0.15703950822353363, 0.8429604768753052], [0.9787266254425049, 0.02127334661781788], [0.9999462366104126, 5.3769341320730746e-05], [0.9998877048492432, 0.0001123320689657703], [0.998832643032074, 0.0011673765257000923], [0.010846600867807865, 0.9891533255577087], [0.9963445067405701, 0.003655546111986041], [0.9998797178268433, 0.00012030315701849759], [0.16472293436527252, 0.8352770209312439], [0.9908350706100464, 0.009164958260953426], [0.9999417066574097, 5.82600332563743e-05], [0.7597764134407043, 0.24022352695465088], [0.03301708400249481, 0.9669829607009888], [0.9840384721755981, 0.0159615408629179], [0.6024317741394043, 0.3975681960582733], [0.9999637603759766, 3.6287088732933626e-05], [0.9794487357139587, 0.02055126056075096], [0.8279911875724792, 0.17200882732868195], [0.9988065958023071, 0.0011934277135878801], [0.5867905020713806, 0.413209468126297], [0.9908086657524109, 0.009191369637846947], [0.9997507929801941, 0.00024924013996496797], [0.9697872996330261, 0.030212657526135445], [0.8771056532859802, 0.122894287109375], [0.9999791383743286, 2.084134212054778e-05], [0.9992577433586121, 0.0007422741036862135], [0.9990946054458618, 0.0009053857065737247], [0.9982370138168335, 0.0017629964277148247], [0.9999668598175049, 3.313086926937103e-05], [0.999313235282898, 0.0006867202464491129], [0.9983075857162476, 0.0016924201045185328], [0.26230308413505554, 0.7376968860626221], [0.9998323917388916, 0.00016760527796577662]]

nth = 1
predictions = [pred for pred in predictions[::nth]]
softmax_probabilities = [prob[1] for prob in softmax[::nth]]

annotations_ambcont = [min(annot, 1) for annot in annotations[::nth]]
annotations_ambno = [annot//2 for annot in annotations[::nth]]

# annotations = annotations_ambno
annotations = annotations_ambcont
print(len(annotations))

def visualize_pie_chart(data, ax, title):
    counts = [data.count(0), data.count(1)]
    ax.pie(counts, labels=["No Contact", "Contact"], autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")

def visualize_bars(data, ax, title, y):
    x = np.arange(len(data))
    colors = ["white" if value == 0 else "red" for value in data]

    for i, (c, value) in enumerate(zip(colors, data)):
        ax.barh(y, value, height=0.5, color=c, linewidth=1, align="edge", left=i)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Predictions", "Contact / No Contact"])
    ax.set_title(title)



# Create figure 1 for the timelines
fig1, ax1 = plt.subplots(figsize=(12, 3))
visualize_bars(annotations, ax1, "Annotations and Predictions", y=0.5)
visualize_bars(predictions, ax1, "", y=0)

# Add black line to split the two timelines
ax1.axhline(y=0.5, color='black', linewidth=1)

fig1.tight_layout()

# Create figure 2 for the pie charts
fig2, (ax2a, ax2b) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
visualize_pie_chart(annotations, ax2a, "Annotations")
visualize_pie_chart(predictions, ax2b, "Predictions")
fig2.tight_layout()

fig3, ax3 = plt.subplots(figsize=(12, 6))

# Plot the annotations as a digital continuous function (step plot)
ax3.step(np.arange(len(annotations)), annotations, color="green", linewidth=2, where='post')

# Plot the softmax probabilities as a smooth continuous function
ax3.plot(np.arange(len(softmax_probabilities)), softmax_probabilities, color="red", linewidth=1)

ax3.set_title("Annotations and Softmax Probabilities")
ax3.set_xlabel("Time")
ax3.set_ylabel("Value")
# ax3.legend(["Softmax Probabilities", "Annotations"], loc='upper right', borderaxespad=0)

fig3.tight_layout()

plt.show()


