#include "DataSheet.hpp"
#include "GBM.hpp"
#include <string>

int main(int argc, char* argv[]) {
	typedef std::string FeaType;
	Parameters param;
	qlib::OptParser<Parameters> opr(param);
	param.opr = &opr;
	if(argc==1) {
		fprintf(stderr,
				"  Forest Booster\n"
				" ================\n"
				"  Usage: %s [conf_file] [options]\n\n", argv[0]);
		opr.print_help();
		return 1;
	}
	//
	if(argv[1][0]!='-') { // interpreted as conf_file
		FILE * conf_file = fopen(argv[1],"r");
		if(not conf_file) {
			fprintf(stderr, "[ERROR] Cannot open '%s' for reading.\n",argv[1]);
			return -1;
		}
		opr.read(conf_file);
		fclose(conf_file);
		// read rest options
		opr.read(argc-2, argv+2);
	} else {
		opr.read(argc-1, argv+1);
	}
	if(opr.check()!=0)
		return -1;
	else {
		fprintf(stderr, "Options:\n");
		opr.print_value();
	}
	FeaTable<FeaType> ft;
	std::vector<double> y;
	GBM<FeaType> gbm(param);
	ft.from_libsvm_mp(param.train_file_path.c_str(), y);
	ft.sort();
	gbm.set_train_data(ft, y);
	gbm.assign_weights(false); // todo: add to parameters
	gbm.boost();
	gbm.save_to_file(param.out_model_path.c_str());
	// save to conf file
	if(param.out_conf_path!="") {
		FILE * new_conf = fopen(param.out_conf_path.c_str(),"w");
		opr.set_file(new_conf);
		opr.print_conf();
		fclose(new_conf);
	}
	//
	return 0;
}

