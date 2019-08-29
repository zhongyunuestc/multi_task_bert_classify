#coding:utf-8
###################################################
# File Name: connect_UNIT.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年05月14日 星期一 17时54分49秒
#=============================================================
import ujson as json
import time
import datetime
import hashlib
import requests
import codecs
import numpy as np







candidates = [{"text":"居住证如何办理呀",
  "uuid_code":"6632",
  "features":[-0.9489043951034546,0.45530375838279724,0.9771518707275391,-0.400733083486557,0.9905833601951599,
        -0.961036741733551,-0.9322264194488525,0.9999077916145325,0.9979562759399414,-0.9960530996322632,
        0.9431678056716919,-0.45015963912010193,0.9933567047119141,-0.24820294976234436,0.9513407945632935,
        -0.2983721196651459,-0.9237732291221619,0.9040918946266174,-0.9934481382369995,0.01831902749836445,
        0.9957407712936401,-0.3050864040851593,0.9013221263885498,-0.9811649322509766,-0.8477751016616821,
        0.7194291353225708,0.43701261281967163,0.9998673796653748,-0.01954258233308792,0.32392817735671997,
        0.5105716586112976,0.6467874050140381,0.9988254308700562,0.129701167345047,0.01876973919570446,
        -0.7902619242668152,0.6363006830215454,-0.9129879474639893,0.8484175205230713,-0.8257334232330322,
        -0.9611008763313293,-0.5257805585861206,0.9629060626029968,0.47456833720207214,0.9050461649894714,
        0.8799929618835449,-0.9402531385421753,-0.33692532777786255,0.944322943687439,0.9987444877624512,
        0.9527807831764221,0.913082480430603,-0.7346636652946472,0.4469316899776459,-0.16378188133239746,
        -0.34373605251312256,-0.9913992285728455,-0.33384811878204346,-0.6626057624816895,-0.8746079206466675,
        0.733983039855957,-0.7606536149978638,0.36415037512779236,-0.9933858513832092,0.21879364550113678,
        -0.8012650012969971,-0.8759597539901733,0.938894510269165,0.4263569116592407,0.579069972038269,
        0.29172366857528687,-0.9840046763420105,0.9805144667625427,0.9998558163642883,-0.8136172890663147,
        0.630605936050415,-0.9730303287506104,0.4029535949230194,-0.4895411431789398,-0.999664843082428,
        0.6825269460678101,-0.02078959532082081,-0.5052513480186462,-0.9377362728118896,-0.8884599804878235,
        -0.8415815830230713,-0.7271517515182495,0.5169693231582642,-0.7118933200836182,-0.012173039838671684,
        -0.8842812180519104,0.7209916114807129,-0.23591077327728271,0.3263294994831085,-0.7141319513320923,
        -0.9463641047477722,-0.13930357992649078,0.5592614412307739,-0.4824788570404053,-0.2589642107486725,
        0.8446611762046814,-0.9864938259124756,-0.4405069053173065,-0.47786349058151245,0.5229969024658203,
        -0.7749990820884705,0.45977309346199036,-0.39119574427604675,-0.3922741711139679,0.9919644594192505,
        -0.9814524054527283,0.9843696355819702,-0.3859853148460388,0.7723616361618042,-0.8429195880889893,
        0.600677490234375,0.9951947331428528,-0.9336833953857422,0.9483498930931091,-0.6586180329322815,
        -0.3121543526649475,0.8716594576835632,-0.9720023274421692,-0.9999542236328125,0.8804137706756592,
        0.9132393598556519,0.22881177067756653,-0.9742834568023682,0.9423383474349976,0.12871940433979034,
        0.8124850988388062,0.63178950548172,0.8265877366065979,0.5151671171188354,-0.9583844542503357,
        0.5255977511405945,0.8091638684272766,0.6168678402900696,0.8442003130912781,0.18339799344539642,
        0.6363542675971985,-0.7693153619766235,-0.7019097208976746,0.18024896085262299,0.9025426506996155,
        -0.9698526263237,-0.24083378911018372,-0.3024076223373413,-0.1758408099412918,0.998688280582428,
        0.9344186782836914,0.961975634098053,-0.506127655506134,-0.2658381462097168,-0.8103361129760742,
        -0.9898971915245056,0.7749605774879456,0.9971142411231995,-0.41487354040145874,-0.6531508564949036,
        -0.5753642320632935,0.9934587478637695,-0.9993120431900024,0.9991530776023865,-0.7265015840530396,
        0.9817311763763428,-0.43676549196243286,0.7997866272926331,0.8437914252281189,0.5985309481620789,
        -0.10781222581863403,0.7374532222747803,0.37570226192474365,0.32887908816337585,-0.9847686886787415,
        0.9970870018005371,-0.9544761776924133,0.9416602253913879,-0.9998096823692322,0.34147319197654724,
        -0.9273940324783325,-0.8066494464874268,0.38712969422340393,-0.9993704557418823,-0.9414555430412292,
        -0.5713714957237244,0.8193821310997009,0.09952159225940704,0.5482757687568665,0.9825255870819092,
        0.8374531269073486,0.9578853845596313,-0.14425364136695862,0.9838998317718506,0.9103294610977173,
        -0.11643840372562408,0.22258330881595612,0.9400586485862732,-0.22062936425209045,-0.9107491970062256,
        -0.9259251952171326,0.5805073976516724,-0.5744878649711609,-0.41179904341697693,-0.9602535367012024,
        -0.2904825210571289,0.366318017244339,0.9632901549339294,0.9860741496086121,-0.7192403674125671,
        0.9726712107658386,0.3663540184497833,-0.7338770627975464,0.2824193239212036,0.9508103728294373,
        0.9903807044029236,0.47738298773765564,-0.1869918256998062,-0.9869012832641602,0.34215453267097473,
        -0.7996448874473572,-0.9543333649635315,-0.09729320555925369,0.06095055490732193,-0.983758807182312,
        -0.4661575257778168,0.4592605531215668,0.8666090369224548,-0.7878090143203735,-0.7630122900009155,
        0.4778340756893158,0.4357491433620453,0.7289457321166992,-0.7464097738265991,-0.9879709482192993,
        0.2980748414993286,-0.9989123940467834,0.5781399607658386,0.1756736934185028,-0.976413369178772,
        0.6702837347984314,-0.9562869071960449,0.9274120926856995,0.6802322864532471,-0.4445561468601227,
        -0.7636176943778992,0.7466889023780823,0.9918355345726013,0.8213205933570862,-0.23966750502586365,
        -0.9998821020126343,-0.8838152289390564,-0.3156511187553406,-0.7302759289741516,-0.6428555250167847,
        -0.9677190184593201,-0.9447794556617737,-0.043857473880052567,-0.7319105267524719,-0.46200117468833923,
        0.9736288189888,-0.9969918727874756,0.687760591506958,0.7104978561401367,-0.3888472020626068,
        0.2550836205482483,0.9223198294639587,-0.9883919954299927,0.9767560958862305,0.8605387210845947,
        -0.9538959264755249,-0.9899100661277771,0.7365246415138245,-0.956807017326355,0.2723880708217621,
        -0.043231431394815445,-0.7581843137741089,-0.9503684043884277,0.19507943093776703,0.703728199005127,
        0.9277276396751404,-0.1084941178560257,0.9896100759506226,0.625371515750885,-0.1266670525074005,
        -0.9541447758674622,-0.8564704060554504,-0.9821285605430603,0.8868309259414673,-0.9592583179473877,
        -0.9988563060760498,-0.22420810163021088,-0.7885667681694031,-0.9981706142425537,0.7765836119651794,
        -0.9657681584358215,-0.8270152807235718,-0.9852232933044434,-0.9718199968338013,-0.9799187779426575,
        -0.9025392532348633,0.9081527590751648,-0.08318710327148438,-0.5195310711860657,0.9318077564239502,
        -0.9435558915138245,0.949057936668396,0.04528312385082245,0.5870431661605835,-0.964954137802124,
        0.9754314422607422,-0.9896301627159119,0.9424172639846802,-0.7107594609260559,0.7612840533256531,
        0.9992690682411194,-0.9895094037055969,-0.4776756167411804,0.18601039052009583,0.4825765788555145,
        -0.7418959736824036,-0.9572226405143738,0.05677004158496857,-0.6824052929878235,-0.9577927589416504,
        0.9068151116371155,0.425313800573349,0.8969454765319824,-0.8752686977386475,0.7686139345169067,
        -0.3602393567562103,0.9649934768676758,0.013165229931473732,0.9986509084701538,-0.7974821329116821,
        -0.8406749367713928,-0.3457682430744171,0.9146108627319336,0.9434748291969299,-0.5434818863868713,
        -0.8861883282661438,0.9727439284324646,0.35763436555862427,-0.12100955098867416,-0.7533837556838989,
        0.8742511868476868,0.9386541843414307,-0.9995133280754089,-0.9541106820106506,0.8234909772872925,
        -0.9155722856521606,0.9875928163528442,-0.04611556977033615,-0.7116373777389526,-0.29178664088249207,
        0.8276073932647705,-0.7010235786437988,-0.7907114028930664,-0.8206971287727356,-0.9647536873817444,
        0.8853092789649963,-0.08699260652065277,-0.649262011051178,-0.8494689464569092,-0.6116626262664795,
        0.9742186069488525,0.5287527441978455,-0.975105881690979,0.012577311135828495,0.728413462638855,
        -0.8860323429107666,-0.979129433631897,0.47881999611854553,-0.9424762725830078,0.6609587073326111,
        0.996169924736023,0.5998165607452393,-0.4374096989631653,-0.033551499247550964,0.1254582554101944,
        0.9516506195068359,-0.385868638753891,-0.9239796996116638,-0.9568552374839783,-0.5282126069068909,
        0.9458721876144409,-0.4564797878265381,0.4510136544704437,-0.770758330821991,-0.7061231136322021,
        -0.9820147156715393,0.18358542025089264,-0.8021263480186462,0.9179916977882385,0.10753997415304184,
        0.1642773151397705,-0.9671405553817749,-0.033571165055036545,0.9866048097610474,0.8871776461601257,
        0.7896057367324829,-0.5518373250961304,-0.5169941782951355,-0.817766547203064,-0.5105622410774231,
        -0.9377861022949219,0.7843571305274963,0.9886766076087952,0.9770957827568054,-0.9171648621559143,
        -0.8619093894958496,-0.9877986907958984,0.8793691992759705,-0.3443732261657715,-0.6813823580741882,
        -0.08469042927026749,-0.32209572196006775,-0.871631920337677,-0.22503544390201569,-0.434918612241745,
        -0.9301397800445557,0.9981257915496826,-0.5200787782669067,0.99607914686203,0.2450295090675354,
        -0.9762759208679199,-0.3375036120414734,0.9968549013137817,0.6296122670173645,0.6263769268989563,
        0.9310228228569031,0.8690397143363953,0.9085667133331299,-0.867432177066803,-0.45314139127731323,
        0.8401567935943604,-0.6817041635513306,0.9844083786010742,0.5064483880996704,0.028438203036785126,
        0.15961600840091705,-0.030528375878930092,-0.5198375582695007,-0.9468618035316467,-0.5556373596191406,
        0.7532653212547302,0.2513340711593628,0.8591583967208862,0.44237905740737915,-0.999605655670166,
        0.6949166655540466,0.9340024590492249,-0.9289411306381226,0.4458985924720764,-0.1189713254570961,
        0.9909825325012207,0.7580288648605347,-0.648223876953125,0.5353667736053467,-0.9805760979652405,
        -0.39325520396232605,0.827223002910614,0.9996733665466309,-0.9331525564193726,0.9081206321716309,
        0.12518762052059174,0.9238568544387817,0.9996656775474548,0.9848057627677917,-0.10573936998844147,
        0.8280761241912842,0.1742425113916397,-0.1275695413351059,-0.8645159006118774,-0.7185624837875366,
        0.8353968262672424,0.8070461750030518,0.9804732203483582,0.9806125164031982,-0.22863708436489105,
        -0.9670429825782776,-0.9833739399909973,-0.7474378347396851,-0.8397879600524902,-0.18188168108463287,
        0.9952817559242249,-0.9047312140464783,-0.6680230498313904,-0.7968794703483582,-0.9543210864067078,
        0.9990167021751404,-0.8764971494674683,-0.9853749871253967,-0.9797451496124268,-0.792565107345581,
        -0.9667876362800598,-0.9581587910652161,0.9023051261901855,-0.9481708407402039,-0.0919996052980423,
        0.9691547751426697,-0.9874786138534546,0.964938759803772,0.5120638608932495,0.6061460971832275,
        -0.3597880005836487,-0.33119651675224304,-0.7570175528526306,-0.8902451992034912,-0.08280544728040695,
        0.7696436047554016,0.9563859105110168,0.9397528767585754,0.7127644419670105,-0.9984135031700134,
        -0.6389056444168091,0.9324167966842651,0.22956135869026184,0.9949545860290527,0.349216103553772,
        -0.9992715716362,0.8789110779762268,-0.09431980550289154,0.5608572363853455,0.9930740594863892,
        0.9443100094795227,-0.5407073497772217,0.629607081413269,-0.0397256538271904,0.7075642943382263,
        0.8186679482460022,-0.6870679259300232,0.8277833461761475,0.6096958518028259,-0.9233178496360779,
        -0.16894255578517914,-0.978485107421875,0.17687590420246124,0.8997476696968079,0.9995850920677185,
        0.8751365542411804,0.098780058324337,-0.9840133786201477,-0.9723283052444458,0.8660090565681458,
        0.9162809252738953,-0.023825369775295258,-0.9463736414909363,0.8916357755661011,-0.6354365348815918,
        0.6241983771324158,-0.9123165607452393,-0.42882707715034485,-0.9343889951705933,0.9227795600891113,
        -0.9547792077064514,0.4546910226345062,-0.8419102430343628,0.9815857410430908,-0.997045636177063,
        0.036630045622587204,0.7380660772323608,0.8038013577461243,0.6186845302581787,0.33936017751693726,
        0.889105498790741,0.491872102022171,0.9520309567451477,-0.9206787943840027,0.9674673080444336,
        0.9417288899421692,-0.9301003217697144,0.7830597758293152,0.7042604684829712,0.10198909789323807,
        0.5307813882827759,0.5868507623672485,0.4010380506515503,0.8266385793685913,-0.823117733001709,
        -0.7118113040924072,-0.8501154780387878,-0.37855854630470276,0.5443711876869202,-0.9935354590415955,
        0.08525124192237854,-0.09936898946762085,-0.8878272771835327,-0.6784363985061646,
        -0.8813589215278625,0.8504707217216492,-0.9882487058639526,0.8542366027832031,-0.7729201316833496,
        0.9714365005493164,0.9874927401542664,-0.858222246170044,0.976516604423523,-0.9988107085227966,
        -0.40972986817359924,-0.9934110045433044,-0.9674900770187378,0.43328914046287537,0.988223671913147,
        -0.22332225739955902,-0.264894038438797,-0.7413027286529541,-0.39789527654647827,
        -0.32210400700569153,0.5153016448020935,-0.9894881844520569,-0.9799178838729858,-0.6333954334259033,
        -0.7158151268959045,0.48467668890953064,-0.8053619861602783,0.5700002312660217,
        -0.6218874454498291,-0.9463263750076294,0.9927861094474792,0.325934499502182,-0.3909878134727478,
        -0.5485363602638245,-0.9983557462692261,-0.9835164546966553,-0.7245117425918579,
        0.9619327783584595,-0.08160359412431717,-0.8088989853858948,-0.15656141936779022,
        -0.9871255159378052,-0.9474031329154968,-0.5552123785018921,0.9962250590324402,-0.006732336711138487,
        0.9063622951507568,0.4887087345123291,0.9507256746292114,0.6988507509231567,-0.6820188760757446,
        -0.0824018269777298,0.7281368374824524,0.8143613338470459,-0.9621537923812866,0.011537865735590458,
        -0.7244787812232971,-0.9776210784912109,-0.0022975087631493807,-0.202842578291893,0.1289282888174057,
        0.9870378971099854,0.6697366237640381,-0.17027701437473297,-0.7854849100112915,-0.9648080468177795,
        0.9336814284324646,0.8541353344917297,-0.9782745242118835,-0.8825699090957642,-0.8903946280479431,
        -0.46404486894607544,0.7046894431114197,0.019203023985028267,0.02507908083498478,-0.9377431273460388,
        -0.43405434489250183,-0.8475465178489685,0.5558784604072571,0.5288100838661194,-0.7600454092025757,
        0.5073754191398621,0.6830824613571167,-0.07199615985155106,0.5332484245300293,
        0.6700762510299683,0.7838609218597412,-0.7129645347595215,0.2655765116214752,0.9888186454772949,
        0.5151128768920898,-0.21272791922092438,-0.8694396615028381,-0.7517410516738892,0.25289881229400635,
        0.9927917718887329,0.9612898230552673,0.9258673787117004,-0.48719799518585205,0.2772952914237976,
        -0.6207532286643982,0.9786987900733948,-0.29959267377853394,0.21382957696914673,0.9370297789573669,
        -0.9359053373336792,0.8984166979789734,0.4363246560096741,0.9899471998214722,-0.2090575397014618,
        0.8557224869728088,-0.977983295917511,0.2204521745443344,0.9731134176254272,0.215925395488739,
        -0.8225612640380859,-0.7749874591827393,0.9677572250366211,-0.41350531578063965,0.9163142442703247,
        0.11878306418657303,0.7584056258201599,-0.401137113571167,0.9999129176139832,0.9790908098220825,
        0.9827452301979065,0.923835813999176,0.9959924221038818,-0.518522322177887,-0.1592319905757904,
        -0.5829684138298035,-0.9900804162025452,0.9931448101997375,0.8306784629821777,-0.9982091188430786,
        -0.562140166759491,0.4907311499118805,0.40453431010246277,-0.739788293838501,-0.2760651111602783,
        0.8892909288406372,0.4202142059803009,0.9407203197479248,0.8262360692024231,-0.41166672110557556,
        0.8549109697341919,0.8240941166877747,0.7323232293128967,0.4524827301502228,-0.4608016014099121,
        -0.6459249258041382,0.9472684860229492,0.7674230337142944,-0.8823089003562927,0.9992008805274963,
        0.6529669165611267,-0.6994050741195679,-0.9560175538063049,-0.027966756373643875,0.9670802354812622,
        0.1314481943845749,-0.9914007186889648,0.43601682782173157,0.6538203358650208,-0.5374933481216431,
        0.03928685933351517,0.6968547701835632,-0.9469227194786072,0.9866107106208801,0.6707606911659241,
        -0.781192421913147,0.5241624116897583,-0.5677193999290466,0.9548525810241699,-0.9965663552284241,
        0.033178236335515976,-0.40317657589912415,0.10714168846607208,0.7520685791969299]}, 
  {"text": "预算周期",
  "uuid_code": "3302",
  "features": [-0.5579898357391357, -0.7924133539199829, -0.8446061015129089, 0.937472939491272, -0.8557257056236267, 0.8865088820457458, 
                -0.7145519852638245, 0.11687327176332474, 0.573927640914917, 0.703902006149292, -0.9029207825660706, -0.9141801595687866,
                -0.6840093731880188, -0.09478640556335449, -0.9863924384117126, 0.1540970355272293, -0.1768144965171814, -0.0674428939819336,
                0.9792649149894714, -0.811119019985199, 0.8738805651664734, -0.9362931847572327, -0.7124188542366028, 0.8251715302467346, 
                0.527147114276886, -0.7062041163444519, -0.8405440449714661, -0.3932020366191864, 0.9879894852638245, -0.9133851528167725, 
                0.8829139471054077, 0.11072244495153427, -0.4671874940395355, 0.9986280798912048, 0.9959909915924072, -0.9627665281295776, 
                -0.01310678943991661, -0.957078218460083, 0.7358953356742859, -0.8879140019416809, -0.9900302886962891, 0.9773315191268921, 
                0.9565017223358154, 0.6177624464035034, 0.9006496667861938, 0.4516626298427582, -0.8018707036972046, -0.6723642945289612, 
                -0.9964424967765808, -0.27915075421333313, -0.24682319164276123, -0.7151771783828735, -0.7134014964103699, 0.10025998204946518, 
                0.9622007012367249, 0.9313629865646362, 0.9032725095748901, 0.13499590754508972, 0.9822328686714172, -0.9094282984733582, 
                -0.9476980566978455, -0.9301507472991943, 0.8100548386573792, 0.996271014213562, -0.7710728645324707, -0.2659922242164612,
                0.8194617629051208, 0.9869565367698669, -0.9931300282478333, 0.3753894567489624, 0.5472344756126404, -0.7763565182685852,
                -0.7075836658477783, 0.6396264433860779, -0.8453917503356934, -0.3656197786331177, -0.9150125980377197, 0.9245871305465698,
                -0.6403997540473938, -0.9986709952354431, 0.34069305658340454, -0.4107718765735626, 0.8630925416946411, -0.31351718306541443,
                0.9873594045639038, -0.5339166522026062, 0.9269359111785889, 0.8887659907341003, -0.12014994770288467, -0.9824631810188293,
                -0.9712812304496765, -0.7790002226829529, -0.6415258049964905, -0.33498522639274597, -0.17768162488937378, 0.9175418019294739,
                0.4712713658809662, -0.9475390315055847, -0.9758579134941101, 0.8057893514633179, 0.8736764788627625, -0.06109487637877464,
                -0.7052949666976929, -0.5455470681190491, 0.9978612661361694, -0.7525861263275146, -0.5169522762298584, 0.5068274140357971,
                -0.8482701182365417, 0.07589216530323029, -0.6435436010360718, 0.9937642216682434, -0.6765062808990479, -0.15341085195541382,
                0.7227129936218262, -0.9505347609519958, 0.9421989917755127, -0.594538152217865, -0.43421685695648193, 0.756317675113678,
                -0.9952179789543152, -0.8832281827926636, 0.6931628584861755, 0.0278791394084692, 0.46762117743492126, -0.459888219833374,
                0.9080841541290283, 0.3572651147842407, -0.8897832036018372, 0.48900091648101807, 0.9768848419189453, 0.5584093928337097,
                -0.6273526549339294, -0.37967348098754883, 0.9796595573425293, 0.786142885684967, 0.38561901450157166, -0.562638521194458,
                -0.6325862407684326, 0.586300253868103, -0.4138723909854889, -0.9249150156974792, -0.6055152416229248, 0.9214903712272644,
                -0.5791521668434143, 0.7540799975395203, 0.123053178191185, 0.9740123748779297, -0.4755878150463104, 0.4338560104370117,
                -0.7555100321769714, 0.17002636194229126, 0.9266114234924316, -0.7418469190597534, 0.9714711904525757, 0.8929241895675659,
                0.9441949725151062, -0.9571418166160583, -0.9975396394729614, 0.23059412837028503, 0.9738861918449402, -0.9919604063034058,
                0.1851194053888321, -0.4295271635055542, -0.9864226579666138, 0.824752151966095, -0.4939139187335968, 0.009977440349757671,
                -0.5284661054611206, -0.4784676730632782, 0.8181383013725281, -0.8308572173118591, -0.9955218434333801, 0.7949443459510803,
                -0.9980310797691345, -0.8629915714263916, -0.9529565572738647, -0.4047867953777313, 0.6857149004936218, 0.933800995349884,
                -0.35148733854293823, -0.6467673182487488, 0.9681586623191833, 0.7510990500450134, 0.2314581274986267, 0.5920889377593994,
                0.6199506521224976, 0.17401562631130219, -0.7976830005645752, 0.8128732442855835, 0.9673884510993958, -0.39745384454727173,
                0.1686454564332962, 0.06079445779323578, -0.2940712571144104, -0.09429781883955002, 0.042746882885694504, -0.703944981098175,
                0.8150107264518738, 0.7484744787216187, -0.7424951195716858, -0.8715064525604248, 0.39085426926612854, 0.9647366404533386,
                0.9905580282211304, 0.14049409329891205, 0.5112552046775818, 0.9961715936660767, -0.30546921491622925, -0.8668577075004578,
                0.954272449016571, 0.9399937391281128, -0.17923134565353394, -0.9797381162643433, 0.9539962410926819, -0.8000150918960571,
                -0.7600581049919128, 0.9717548489570618, 0.9985469579696655, -0.735304594039917, -0.876198410987854, 0.9637479186058044,
                -0.7399601936340332, -0.3852103054523468, -0.1500021517276764, -0.048249129205942154, 0.16023093461990356, -0.8281475305557251,
                0.7754027843475342, 0.5905630588531494, -0.6647814512252808, -0.8927069902420044, -0.18548069894313812, 0.5761966109275818,
                -0.7911154627799988, 0.9530069828033447, 0.9227659106254578, 0.3684873878955841, 0.8023464679718018, 0.9321893453598022,
                0.08985746651887894, 0.9969925880432129, 0.830978512763977, 0.21325644850730896, 0.07888754457235336, -0.6238937973976135,
                0.908596396446228, -0.2936044931411743, -0.8738970756530762, -0.47875291109085083, 0.4819149076938629, -0.8984719514846802,
                0.9665868282318115, 0.2754856050014496, -0.1690010130405426, -0.7588514089584351, 0.5279146432876587, 0.8950427770614624,
                0.6748839616775513, -0.49227380752563477, -0.9726483225822449, 0.6399024128913879, -0.06193648278713226, -0.16513213515281677, 
                -0.28503847122192383, -0.1280481219291687, 0.828281581401825, 0.41626834869384766, 0.05723629519343376, -0.9849716424942017, 
                -0.9800552129745483, -0.6720156073570251, 0.9886629581451416, 0.8741400837898254, -0.6194972991943359, -0.46559593081474304,
                0.999224841594696, -0.7645136117935181, -0.43732506036758423, -0.29214438796043396, 0.962979257106781, 0.9960345029830933, 
                0.8502422571182251, -0.6987284421920776, 0.3808126449584961, 0.21160344779491425, -0.6393829584121704, 0.062224436551332474,
                0.29357147216796875, 0.32987791299819946, -0.8281463980674744, 0.8334084153175354, 0.7881921529769897, -0.913662314414978,
                0.6405017375946045, -0.6102097034454346, 0.5148483514785767, 0.9384191036224365, 0.4659920632839203, 0.9870499968528748,
                0.1424025297164917, 0.7787440419197083, 0.7673333883285522, 0.06901765614748001, 0.36302676796913147, -0.6246880888938904, 
                0.9909111261367798, -0.9957535266876221, -0.7080358266830444, -0.8686379194259644, 0.7306337356567383, 0.02244022861123085,
                -0.38870909810066223, 0.9633026719093323, -0.9562495350837708, -0.8676162362098694, -0.11328784376382828, -0.9633893370628357,
                -0.09601852297782898, -0.9320275187492371, 0.9936834573745728, 0.993621289730072, -0.014894332736730576, 0.5363871455192566,
                -0.34221240878105164, 0.7312833070755005, -0.881375253200531, 0.9771977663040161, 0.04213009402155876, 0.9991199374198914,
                -0.9867562055587769, -0.8027236461639404, 0.9302317500114441, -0.02697698585689068, -0.9679551720619202, 0.725771963596344,
                0.6286370158195496, -0.9663263559341431, 0.3899708688259125, 0.9864837527275085, 0.19074946641921997, 0.67046058177948, 
                0.3420935273170471, 0.990509033203125, 0.7617543339729309, 0.4447328746318817, -0.9944847226142883, 0.9515237212181091, 
                -0.7406182885169983, 0.8838799595832825, 0.9260266423225403, -0.3966374099254608, -0.9605501294136047, 0.9855712056159973,
                0.9983893632888794, -0.8406240344047546, 0.8602049946784973, 0.7233443260192871, 0.978293240070343, 0.8381167054176331,
                0.8514677882194519, -0.949465811252594, -0.19438983500003815, -0.15994271636009216, -0.3971148431301117, -0.8308188319206238,
                0.9426986575126648, -0.3925280272960663, 0.9692785739898682, 0.9913989901542664, -0.33024612069129944, 0.5359615683555603,
                -0.9917538166046143, -0.9768420457839966, -0.9791557192802429, -0.9969528913497925, -0.9726291298866272, 0.6346064209938049,
                0.9903666377067566, 0.9071061611175537, -0.9110742211341858, -0.6584275364875793, 0.9297432899475098, 0.9552720189094543, 
                0.9412879943847656, 0.6226886510848999, -0.7984674572944641, 0.9924956560134888, -0.8846539258956909, -0.8778584003448486, 
                0.8905429244041443, -0.16052894294261932, 0.4376327395439148, 0.5342604517936707, 0.9952842593193054, 0.5271583795547485, 
                0.8970943093299866, 0.9538596272468567, -0.4704766571521759, 0.9836394190788269, -0.9362856149673462, -0.6335283517837524,
                -0.9696645140647888, 0.9662428498268127, 0.9228832125663757, -0.02875048667192459, 0.7723910212516785, 0.9702293276786804,
                0.7130836844444275, 0.7010065913200378, -0.3182694613933563, -0.7959766983985901, -0.9317563772201538, 0.8509432077407837, 
                0.6028692126274109, 0.7886121273040771, 0.372391939163208, -0.9863549470901489, 0.4806936979293823, 0.9794929027557373,
                -0.15218764543533325, 0.7435689568519592, 0.3563397228717804, 0.4530564248561859, 0.8615255355834961, 0.3129073977470398,
                0.9348183870315552, -0.9159221053123474, -0.9618852138519287, -0.7562565207481384, 0.9474115371704102, -0.9915899038314819,
                0.2681950330734253, -0.6807636618614197, -0.759729266166687, 0.9570447206497192, -0.9397611618041992, -0.6467227339744568, 
                -0.7228676676750183, 0.8984537124633789, 0.7315589785575867, 0.42056789994239807, 0.7655977606773376, 0.8512859344482422, 
                0.7795860767364502, 0.4497895836830139, -0.8165807127952576, 0.7112129926681519, 0.9206638932228088, -0.9974760413169861, 
                0.18054266273975372, -0.9553656578063965, -0.4041939079761505, 0.5461282730102539, 0.502244234085083, 0.6816388368606567, 
                0.646345317363739, 0.8496853709220886, -0.9902645349502563, -0.8393821716308594, -0.9869637489318848, 0.9653250575065613, 
                -0.8486524224281311, 0.978135347366333, 0.5011361241340637, 0.9736540913581848, -0.7792921662330627, 0.9456872344017029, 
                0.9985707402229309, 0.8681710362434387, 0.15132182836532593, -0.8644813895225525, 0.4709317088127136, -0.45478954911231995,
                0.8189760446548462, -0.13586926460266113, -0.5790030360221863, 0.989662766456604, -0.5040900707244873, -0.8445579409599304,
                -0.8986750245094299, -0.8421037197113037, 0.9627828598022461, -0.2548799216747284, 0.7699987292289734, 0.5351479649543762, 
                0.4795120060443878, -0.9493061900138855, 0.9921499490737915, -0.5101048350334167, -0.8540425896644592, -0.9061245322227478,
                -0.635572075843811, -0.7525273561477661, -0.38255104422569275, 0.9309048652648926, 0.49873822927474976, 0.984348475933075, 
                -0.6278051137924194, -0.4221916198730469, -0.4006151258945465, 0.9569612145423889, -0.9726950526237488, -0.9220481514930725,
                -0.7735413312911987, 0.6027602553367615, -0.992186963558197, -0.8935434818267822, -0.9816449880599976, -0.9586738348007202,
                -0.8810635805130005, -0.391195148229599, -0.4272784888744354, 0.24306471645832062, 0.20789819955825806, -0.885751485824585,
                0.42874231934547424, 0.9988155961036682, -0.029102355241775513, 0.9437161087989807, 0.8572251200675964, 0.6717868447303772, 
                0.797106921672821, 0.3410598337650299, -0.3545558452606201, -0.7271349430084229, 0.9536885619163513, -0.305677592754364,
                -0.29903823137283325, -0.13020697236061096, 0.9228704571723938, 0.989454984664917, 0.8589195013046265, -0.024010995402932167,
                0.9645529985427856, 0.2996836006641388, 0.5033479928970337, -0.43783554434776306, -0.982588529586792, -0.8700205683708191, 
                0.34137654304504395, -0.9411080479621887, -0.6554872393608093, -0.888783872127533, -0.44161203503608704, -0.09806779026985168,
                0.9201487302780151, 0.9968195557594299, 0.8082852959632874, -0.041015900671482086, 0.4746183156967163, 0.9140796065330505,
                -0.9966480135917664, -0.9226406216621399, 0.8974335789680481, 0.8472357988357544, -0.922015368938446, -0.3205150067806244, 
                -0.8012820482254028, -0.9398528933525085, 0.0430181547999382, 0.9188498258590698, -0.806588351726532, -0.999398946762085, 
                -0.8399452567100525, -0.20764334499835968, 0.08019114285707474, -0.8113056421279907, 0.21028336882591248, -0.9935329556465149,
                0.414257675409317, -0.8297836780548096, -0.016453305259346962, -0.46727481484413147, -0.9148938059806824, -0.9854685664176941, 
                -0.22565001249313354, 0.7839601039886475, -0.726323127746582, 0.6473280191421509, -0.5410526394844055, -0.7196989059448242, 
                -0.9828994274139404, -0.3598138391971588, -0.895318329334259, 0.1544635146856308, -0.43392035365104675, 0.7598063945770264,
                0.698718249797821, 0.21477767825126648, -0.6759029626846313, 0.8715725541114807, -0.40676698088645935, 0.9909208416938782, 
                -0.535858154296875, -0.90255206823349, -0.1516890972852707, -0.9723115563392639, 0.9940709471702576, -0.14273850619792938, 
                0.9180236458778381, 0.8595446944236755, -0.716221034526825, 0.9023247361183167, -0.9844049215316772, -0.4421117901802063, 
                -0.8292071223258972, 0.9383000731468201, 0.2771671712398529, -0.898920476436615, 0.9994663596153259, 0.9463731646537781, 
                0.9279355406761169, 0.9709234237670898, 0.4617849290370941, 0.9605165719985962, 0.22540439665317535, -0.19972865283489227, 
                -0.6459276676177979, -0.7165435552597046, -0.903813898563385, 0.4257207214832306, 0.36619070172309875, -0.08719713240861893,
                0.5744730234146118, 0.9700295925140381, 0.9955950975418091, -0.9913657307624817, 0.5297917127609253, 0.4933261573314667, 
                0.7370173335075378, -0.5634458065032959, -0.9991925358772278, -0.24246951937675476, 0.7874792814254761, 0.9465132355690002, 
                -0.8947716355323792, -0.9978055953979492, -0.9157575368881226, -0.8226816654205322, -0.9535903334617615, -0.9942506551742554,
                0.9566786289215088, 0.6848069429397583, 0.8431218862533569, 0.8487921953201294, 0.8748887181282043, 0.641546905040741, 
                -0.8776810765266418, -0.968035876750946, -0.8327024579048157, -0.7352150678634644, -0.8396498560905457, 0.807857871055603,
                0.9391341805458069, 0.4083877503871918, -0.47650638222694397, -0.219225212931633, -0.9843279719352722, 0.7461519241333008, 
                0.7440766096115112, 0.8691077828407288, 0.27058130502700806, 0.6836993098258972, -0.9281202554702759, -0.9860690236091614, 
                -0.8884078860282898, -0.683788537979126, -0.6347919702529907, -0.8865664005279541, 0.8105202317237854, 0.7708137035369873, 
                0.36096346378326416, -0.8674959540367126, -0.8811913728713989, -0.15187834203243256, -0.584209680557251, 0.6748296022415161,
                0.15815137326717377, -0.33987706899642944, 0.9987841248512268, 0.6958291530609131, 0.5235489010810852, -0.795581042766571, 
                0.7266318798065186, 0.9948970675468445, -0.06072939559817314, -0.9136044979095459, -0.9748850464820862, 0.6093827486038208,
                0.7486396431922913, -0.8833609819412231, -0.6246137022972107, 0.5796758532524109, -0.9370136857032776, -0.4599105417728424,
                -0.38666391372680664, 0.6265060305595398, -0.9994081854820251, 0.3318355977535248, 0.8930772542953491, 0.2550109922885895, 
                -0.5018227696418762, -0.7859693169593811, -0.7983625531196594, -0.6041072607040405, -0.4367479085922241, -0.9489052891731262,
                0.9923204183578491, -0.8799620866775513, -0.9795944094657898, 0.5052676200866699, -0.9880563020706177, 0.5763091444969177,
                -0.9920310974121094, 0.8214771747589111, 0.9760679602622986, -0.9535461664199829, 0.9411728382110596, -0.8679402470588684,
                -0.08900630474090576, -0.7361579537391663, 0.30981019139289856, 0.7329975962638855, -0.5671846866607666, 0.9986215829849243,
                0.10078606009483337, -0.9537758827209473, -0.9046668410301208, 0.8550004959106445, 0.21632423996925354, -0.9748008847236633, 
                -0.9436018466949463, 0.8330948948860168, 0.928071141242981, 0.7519788146018982, -0.6603317856788635, 0.9944392442703247, 
                0.8034263849258423, 0.4885670840740204, -0.9919502139091492, -0.985799252986908, -0.9812246561050415, 0.8010563254356384, 
                0.4489594101905823, -0.6683244705200195, 0.6638675332069397, -0.03603596240282059, -0.8672166466712952, -0.9394145607948303, 
                -0.7128564715385437, -0.9959260821342468, -0.9215661287307739, 0.9940155148506165, 0.7867967486381531, -0.8022255897521973,
                0.943392813205719, 0.037993647158145905, -0.779925525188446, 0.2468445599079132, 0.6433215141296387, -0.6419448852539062, 
                0.9821656942367554, -0.3166353702545166, -0.985343873500824, -0.6104264855384827, 0.9673206806182861, 0.9859031438827515]}]




def read_feature_dict(input_file):
    cand2features = {}
    with codecs.open(input_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            line_info = line.split('\t')
            text = line_info[0]
            features = json.loads(line_info[1])['features']
            cand2features[text] = features
    cand2features = {key: np.array(value).astype(float) for key, value in cand2features.items()}
    return cand2features

def gen_candidate_tfidf_model(candidates, tokenizer):
    #corpus 
    splited_candidates = [tokenizer.tokenize(item) for item in candidates]
    tfidf_dict = gensim.corpora.Dictionary(splited_candidates)
    corpus = [tfidf_dict.doc2bow(text) for text in splited_candidates]
    #tfidf
    tfidf_model = gensim.models.TfidfModel(corpus)

    #sim_indices
    corpus_tfidf = tfidf_model[corpus]
    tfidf_sim_indices= gensim.similarities.MatrixSimilarity(corpus_tfidf)
    print(tfidf_model)

    #exit()
    return tfidf_dict, tfidf_model, tfidf_sim_indices




cand2features = read_feature_dict('../features/features.tsv')
candidates = [{'text': text, 'userId': 1, 'features': list(cand2features[text])} for text in cand2features]
#candidates = [{'uuid_code': 1, 'features': list(cand2features[text])} for text in cand2features]
#print(candidates)
#exit()


def http_post_request():
    #url = 'http://47.96.228.29:17123/personranking'
    url = 'http://192.168.7.233:17123/personranking'
    post_data = {}
    with open('query.txt', 'r') as fr:
        for line in fr:
            line = line.strip().lower()
            line_info = line.split('\t')
            post_data['query'] = line_info[0]
            post_data['method'] = 'ranking'
            post_data['candidates'] = candidates

            json_data = post_data
            start_time = datetime.datetime.now()
            results = requests.post(url, json=json_data)
            #results = requests.post(url, data=json_data)
            #print(results.text)
            if (results):
                end_time = datetime.datetime.now()
                rs = json.loads(results.text)
                #end_time = datetime.datetime.now()
                cost = (end_time - start_time).total_seconds() * 1000
                #print(line_info[0] + '\t' + rs[0]['title'] + '\t'  + str(rs[0]['score']) + '\t' + str(cost) + 'ms')
                print(line_info[0] + '\t' + str(rs[0]['score']) + '\t' + str(cost) + 'ms')


    

if __name__ == '__main__':
    print(datetime.datetime.now())
    #http_get_request()
    http_post_request()


    print(datetime.datetime.now())
    #segment_test('我要请假')