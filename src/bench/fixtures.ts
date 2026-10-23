// Bench fixtures; licences and sources are in fixtures/README.md.
import enUrl from './fixtures/en-librispeech.flac?url';
import hiUrl from './fixtures/hi-fleurs.flac?url';

export interface Fixture {
  key: 'en' | 'hi';
  url: string;
  language: 'en' | 'hi';
  /** What was said, as published with the dataset. */
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
];
