#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3L.h"
#include "Utf.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader : public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext() {

		m_instance.clear();
		vector<string> vecLine;
		while (1) {
			string strLine;
			if (!my_getline(m_inf, strLine)) {
				break;
			}
			if (strLine.empty())
				break;
			vecLine.push_back(strLine);
		}

		if (vecLine.size() == 0)
			return NULL;

		vector<string> vecInfo;
		if (vecLine.size() >= 1) {
			split_bychars(vecLine[0], vecInfo, "\t");
			m_instance.m_label = vecInfo[0];
			split_bychar(vecInfo[1], m_instance.m_words, ' ');
			int word_num = m_instance.m_words.size();
			for(int idx = 0; idx < word_num; idx++) {
				m_instance.m_words[idx] = normalize_to_lowerwithdigit(m_instance.m_words[idx]);
			}
		}

		if(vecLine.size() >= 2)
			split_bychar(vecLine[1], m_instance.m_sparse_feats, ' ');

		return &m_instance;
	}

};

#endif

