// Bench fixtures; licences and sources are in fixtures/README.md.
import enUrl from './fixtures/en-librispeech.flac?url';
import enVoxpopuliUrl from './fixtures/en-voxpopuli.flac?url';
import frDeUrl from './fixtures/fr-de-fleurs.flac?url';
import hiUrl from './fixtures/hi-fleurs.flac?url';
import nonspeechUrl from './fixtures/nonspeech-musan.flac?url';

export interface Fixture {
  key: 'en' | 'hi' | 'en-vp' | 'fr-de' | 'nonspeech';
  url: string;
  /** What is spoken. Captions should show only English; anything else should show nothing. */
  language: 'en' | 'hi' | 'fr+de' | 'none';
  /** What was said, as published with the dataset (empty for no speech). */
  reference: string;
  /** Reference English translation (FLoRes), for judging translate output by hand. */
  translationReference?: string;
}

export const FIXTURES: readonly Fixture[] = [
  {
    key: 'en',
    url: enUrl,
    language: 'en',
    reference: [
      'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL',
      "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER",
      'HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND',
      "HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS REALLY GREEK AFTER ALL AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHACA",
      'IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS THESE CRITICISMS ARE HOW DELICATE IN EXPRESSION',
      'ON THE GENERAL PRINCIPLES OF ART MISTER QUILTER WRITES WITH EQUAL LUCIDITY',
    ].join(' '),
  },
  {
    key: 'hi',
    url: hiUrl,
    language: 'hi',
    reference: [
      'वैज्ञानिकों का मानना है कि ओसेलोट्स गंध के द्वारा जानवरों का पीछा करते और खाते (शिकार) हैं, यह सूँघते हुए कि वे ज़मीन पर कहाँ रहे होंगे।',
      'कुतूहल गाँव में आधे घंटे टहलना समय की हानि नहीं है।',
      'यदि आप सर्दियों में आर्कटिक या अंटार्कटिक क्षेत्रों में जाते हैं, तो आप ध्रुवीय रात का अनुभव करेंगे, जिसका अर्थ है कि सूरज क्षितिज से ऊपर नहीं उगता है।',
      'इतालवी भाषा में उच्चारण करना तुलनात्मक रूप से आसान है क्योंकि ज़्यादातर शब्दों का उच्चारण वैसे ही किया जाता है जैसे उन्हें लिखा जाता है',
      'चर्च के परिसर में, कुछ कब्रों पर कबूतरों की संगमरमर से बनाई गयी दिलचस्प मूर्तियां हैं।',
    ].join(' '),
    translationReference: [
      "Scientists think that ocelots follow and find animals to eat (prey) by smell, sniffing for where they've been on the ground.",
      "It's worth half an hour to stroll about the intriguing village.",
      "If you visit the Arctic or Antarctic areas in the winter you will experience the polar night, which means that the sun doesn't rise above the horizon.",
      'Pronunciation is relatively easy in Italian since most words are pronounced exactly how they are written',
      'In the churchyard, there are interesting marble sculptures of doves over some tombs.',
    ].join(' '),
  },
  {
    // Continuous parliamentary speech by four speakers (two of them women), most of them not
    // native English speakers: long runs without a pause, so the segmenter makes real forced cuts.
    key: 'en-vp',
    url: enVoxpopuliUrl,
    language: 'en',
    reference: [
      'It is the corrupt elites who are depriving the people of Angola of their basic human rights access to healthcare, food and education by stealing public money intended for basic social services. We are seriously concerned about the possible mismanagement of EU development funds by the local authorities, and in this regard we are asking the Commission to review the national indicative programme under the 11th EDF and to ensure that good governance, respect for human rights and the fight against corruption are the main financing priorities for this fabulously rich country, which suffers also from a fabulously corrupt elite.',
      'But even worse than those billions is the frustration and anger when a tourist, an artist and we love them, we proved this today a company proprietor or a worker, a student, a patient or a doctor cannot find that necessary information, such as a form to fill out when he or she encounters problems when delivering documents online. This portal will help to solve those problems. I find it especially important today, when some politicians call Europe an imaginary community,',
      "As Members of the European Parliament the only directly elected EU institution we have a crucial role to play in promoting the aims of the European Year of Citizens 2013 by bridging the gap between Brussels and Member States, communicating the rights and benefits of EU citizenship, endorsing citizens' concerns in formulating EU policies and promoting participative democracy.",
      'This resolution tackles the call on the fight against Boko Haram and its terrorism, but also addresses the Nigerian authorities, calling for reform of the security forces, to make sure that they prevent extrajudicial executions, arbitrary detentions and torture, and reform of the criminal judicial system to make sure that the law is enforced to fight violence, terrorism and corruption, which is massive in Nigeria.',
    ].join(' '),
  },
  {
    // Not English: three French then three German sentences. Captions should stay empty.
    key: 'fr-de',
    url: frDeUrl,
    language: 'fr+de',
    reference: [
      "Auparavant, le PDG de Ring, Jamie Siminoff, a fait remarquer qu'il a lancé l'entreprise lorsque la sonnette de sa porte d'entrée n'était pas audible depuis son magasin dans son garage.",
      "Le patient s'était rendu au Nigeria, où certains cas de virus Ebola se sont produits.",
      'Le ski est une activité de voyage majeure qui compte de nombreux aficionados, parfois appelés « ski bums », qui organisent leurs vacances entières en fonction du ski dans des endroits spécifiques.',
      'Die Wissenschaft weist nun darauf hin, dass diese massive Kohlenstoffwirtschaft die Biosphäre aus einem ihrer stabilen Zustände, der die menschliche Evolution in den letzten zwei Millionen Jahren getragen hat, herausgebracht hat.',
      'Der Amazonas ist auch der breiteste Fluss der Erde, teilweise bis zu sechs Meilen breit.',
      'Die lokalen Behörden ermahnen die Einwohner in der Nähe der Anlage dazu, sich innerhalb von Gebäuden aufzuhaltem, Klimaanlagen auszuschalten und kein Leitungswasser zu trinken.',
    ].join(' '),
  },
  {
    // No speech: instrumental music, then digital silence, then background noise. Captions should
    // stay empty; this is where Whisper hallucinates ("Thank you.") and loops.
    key: 'nonspeech',
    url: nonspeechUrl,
    language: 'none',
    reference: '',
  },
];
