/**
 * @file: qopt.hpp
 * @brief: Command-Line Options Parser
 * @author: Qing Wang
 * @date: Aug 19, 2015
 */
#pragma once

#ifndef _QOPT_HPP
#define _QOPT_HPP

#include <string>
#include <iostream>
#include <iomanip> // std::setw
#include <sstream>
#include <map>

namespace qlib {

enum OptStatus {
	NOT_SET, SET, OVERWRITTEN
};
enum OptAction {
	INIT, READ, CHECK, PRINT_HELP, PRINT_VALUE, PRINT_FILE, PRINT_CONF, PRINT_STATUS
};

const char * get_status_name(OptStatus status) {
	switch(status) {
		case NOT_SET:
			return "NOT_SET";
		case SET:
			return "SET";
		case OVERWRITTEN:
			return "OVERWRITTEN";
		default:
			return "Unknown";
	}
}

template<class P>
class OptParser {
	P& opts; // parameter structure
	std::map<std::string,OptStatus> opt_status;
	std::string cur_name;
	std::string cur_input;
	FILE * fo; // for opr_print_file()
	int opr_ret;
	int max_name_len;

	// Init
	template<typename T>
	void opr_init(T& var, const std::string& name, const std::string& descr) {
		if(opt_status.count(name)>0)
			std::cerr << "Option conflict: " << name << std::endl;
		opt_status[name] = NOT_SET;
		max_name_len = max_name_len>name.length()?max_name_len:name.length();
	}
	template<typename T>
	void opr_init(T& var, const std::string& name, const std::string& descr, const T& dval) {
		var = dval;
		return opr_init(var,name,descr);
	}

	// Read	
	template<typename T>
	int opr_read(T& var, const std::string& name, const std::string& descr) {
		if(name==cur_name) {
			std::istringstream iss(cur_input);
			iss >> var;
			switch(opt_status[name]) {
				case NOT_SET:
					opt_status[name] = SET; break;
				case SET:
					opt_status[name] = OVERWRITTEN; break;
				case OVERWRITTEN:
					opt_status[name] = OVERWRITTEN; break;
			}
			return 1;
		}
		return 0;
	}
	template<typename T>
	int opr_read(T& var, const std::string& name, const std::string& descr, const T& dval) {
		return opr_read(var, name, descr);
	}

	// Check
	template<typename T>
	int opr_check(T& var, const std::string& name, const std::string& descr) {
		if(opt_status[name]==NOT_SET) {
			std::cerr << "Option value for " << name << " is required.\n";
			return -1;
		}
		return 0;
	}
	template<typename T>
	int opr_check(T& var, const std::string& name, const std::string& descr, const T& dval) {
		return 0;
	}
	
	// Print Help
	template<typename T>
	void opr_print_help(T& var, const std::string& name, const std::string& descr) {
		std::cerr << std::setw(max_name_len+2) << name << ": " << descr << std::endl;
	}
	template<typename T>
	void opr_print_help(T& var, const std::string& name, const std::string& descr, const T& dval) {
		std::cerr << std::setw(max_name_len+2) << name << ": " << descr << " [" << dval << "]" << std::endl;
	}

	// Print Value
	template<typename T>
	void opr_print_value(T& var, const std::string& name, const std::string& descr) {
		std::cerr << std::setw(max_name_len+2) << name << ": " << var << std::endl;
	}
	template<typename T>
	void opr_print_value(T& var, const std::string& name, const std::string& descr, const T& dval) {
		return opr_print_value(var,name,descr);
	}

	// Print to file
	template<typename T>
	void opr_print_file(T& var, const std::string& name, const std::string& descr) {
		fprintf(fo, "#%s=%s\n",name.c_str()+1,// skip '-'
				qlib::to_string(var).c_str());
	}
	template<typename T>
	void opr_print_file(T& var, const std::string& name, const std::string& descr, const T& dval) {
		if(var!=dval)
			return opr_print_file(var,name,descr);
	}

	// Print to Conf
	template<typename T>
	void opr_print_conf(T& var, const std::string& name, const std::string& descr) {
		fprintf(fo, "# %s\n" "%s=%s\n",descr.c_str(), name.c_str()+1,// skip '-'
				qlib::to_string(var).c_str());
	}
	template<typename T>
	void opr_print_conf(T& var, const std::string& name, const std::string& descr, const T& dval) {
		fprintf(fo, "# %s [%s]\n" "%s%s=%s\n",descr.c_str(),
				qlib::to_string(dval).c_str(),
				var==dval?"#":"", name.c_str()+1,// skip '-'
				qlib::to_string(var).c_str());
	}

	// Print Status
	template<typename T>
	void opr_print_status(T& var, const std::string& name, const std::string& descr) {
		std::cerr << name << ": " << get_status_name(opt_status[name]) << std::endl;
	}
	template<typename T>
	void opr_print_status(T& var, const std::string& name, const std::string& descr, const T& dval) {
		return opr_print_status(var,name,descr);
	}

public:
	/** Class Constructor 
	 */
	OptParser(P& _p): opts(_p),max_name_len(0) { opts(*this,INIT); }

	/** Main facility of OptParser
	 *
	 * This function should not be called directly.
	 * For usage please refer to examples.
	 */
	template<typename T>
	void operator()(OptAction action, T& var, const std::string& name, const std::string& descr, const T& dval) {
		switch(action) {
			case INIT:
				opr_init(var,name,descr,dval); break;
			case READ:
				opr_ret += opr_read(var,name,descr,dval); break;
			case CHECK:
				opr_ret += opr_check(var,name,descr,dval); break;
			case PRINT_HELP:
				opr_print_help(var,name,descr,dval); break;
			case PRINT_VALUE:
				opr_print_value(var,name,descr,dval); break;
			case PRINT_FILE:
				opr_print_file(var,name,descr,dval); break;
			case PRINT_CONF:
				opr_print_conf(var,name,descr,dval); break;
			case PRINT_STATUS:
				opr_print_status(var,name,descr,dval); break;
			default:
				std::cerr << "Unknown action type ["<<action<<"]"<<std::endl;
		}
	}
	/** Main facility of OptParser
	 *
	 * This function should not be called directly.
	 * For usage please refer to examples.
	 */
	template<typename T>
	void operator()(OptAction action, T& var, const std::string& name, const std::string& descr) {
		switch(action) {
			case INIT:
				opr_init(var,name,descr); break;
			case READ:
				opr_ret += opr_read(var,name,descr); break;
			case CHECK:
				opr_ret += opr_check(var,name,descr); break;
			case PRINT_HELP:
				opr_print_help(var,name,descr); break;
			case PRINT_VALUE:
				opr_print_value(var,name,descr); break;
			case PRINT_FILE:
				opr_print_file(var,name,descr); break;
			case PRINT_CONF:
				opr_print_conf(var,name,descr); break;
			case PRINT_STATUS:
				opr_print_status(var,name,descr); break;
			default:
				std::cerr << "Unknown action type ["<<action<<"]"<<std::endl;
		}
	}
	// Special definition for T==std::string
	void operator()(OptAction action, std::string& var, const std::string& name, const std::string& descr, const char dval[]) {
		return (*this)(action,var,name,descr,std::string(dval));
	}

	/** Read options in arguments
	 *
	 * @param len   Number of args
	 * @param args  Arguments
	 *
	 * @return      0 normally, or -1 indicating error.
	 */
	int read(int len, char* args[]) {
		if(len%2!=0) {
			std::cerr << "Wrong number of argument." << std::endl;
			return -1;
		}
		int i=0;
		while(i<len-1) {
			cur_name = std::string(args[i++]);
			cur_input = std::string(args[i++]);
			opr_ret = 0;
			opts(*this,READ);
			if(opr_ret==0) {
				std::cerr << "Cannot find option " << cur_name << std::endl;
				return -1;
			}
			if(opr_ret>1) {
				std::cerr << "Read by multiple options " << cur_name << std::endl;
				return -1;
			}
		}
		return 0;
	}

	/** Read options in configure file
	 *
	 * @param f     opened file.
	 *
	 * @return      0 normally, or -1 indicating error.
	 */
	int read(FILE * f) {
		char * buf = nullptr;
		size_t len = 0;
		while(getline(&buf,&len,f)>0) {
			char * sharp = strchr(buf,'#');
			if(sharp)
				*sharp = '\0';
			char * eq = strchr(buf,'=');
			if(not eq)
				continue;
			std::istringstream(std::string(buf, eq)) >> cur_name;
			cur_name = std::string("-")+cur_name;
			std::istringstream(std::string(eq+1)) >> cur_input;
			opr_ret = 0;
			opts(*this,READ);
			if(opr_ret==0) {
				std::cerr << "Cannot find option " << cur_name << std::endl;
				return -1;
			}
			if(opr_ret>1) {
				std::cerr << "Read by multiple options " << cur_name << std::endl;
				return -1;
			}
		}
		return 0;
	}

	/** Check if opts are filled
	 *
	 * @return 0 normally, or -1 indicating error.
	 */
	int check() {
		opr_ret = 0;
		opts(*this,CHECK);
		return opr_ret==0?0:-1;
	}

	/** Print Help
	 */
	void print_help() {
		opts(*this,PRINT_HELP);
	}

	/** Print Value
	 */
	void print_value() {
		opts(*this,PRINT_VALUE);
	}

	/** Print to File
	 */
	void print_file() {
		opts(*this,PRINT_FILE);
	}

	/** Print to Conf
	 */
	void print_conf() {
		opts(*this,PRINT_CONF);
	}

	/** Set output file (for print_file() and print_conf())
	 */
	void set_file(FILE * f) {
		fo = f;
	}

	/** Print Status
	 */
	void print_status() {
		opts(*this,PRINT_STATUS);
	}
};

}; // namespace qlib
#endif // _QOPT_HPP_

