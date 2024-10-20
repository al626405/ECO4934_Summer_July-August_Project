-- Create the Train table
CREATE TABLE Train (
who INTEGER,
current_resolution_FIXED INTEGER,
current_status_RESOLVED INTEGER,
current_status_VERIFIED INTEGER,
priority_P1 INTEGER,
priority_P2 INTEGER,
priority_P3 INTEGER,
priority_P4 INTEGER,
priority_P5 INTEGER,
product_CDT INTEGER,
product_JDT INTEGER,
product_Platform INTEGER,
version_0_DD_1_0 INTEGER,
version_0_DD_1_1 INTEGER,
version_0_1_3 INTEGER,
version_0_5 INTEGER,
version_0_7 INTEGER,
version_0_7_1 INTEGER,
version_0_8 INTEGER,
version_0_9 INTEGER,
version_0_9_2 INTEGER,
version_1_0 INTEGER,
version_1_0_0 INTEGER,
version_1_0_1 INTEGER,
version_1_0_2 INTEGER,
version_1_0_3 INTEGER,
version_1_1 INTEGER,
version_1_2 INTEGER,
version_1_2_1 INTEGER,
version_1_3_0 INTEGER,
version_1_3_1 INTEGER,
version_1_5 INTEGER,
version_1_5_2 INTEGER,
version_1_5_3 INTEGER,
version_1_7 INTEGER,
version_2_0 INTEGER,
version_2_0_0 INTEGER,
version_2_0_1 INTEGER,
version_2_0_2 INTEGER,
version_2_0_3 INTEGER,
version_2_1 INTEGER,
version_2_1_0 INTEGER,
version_2_1_1 INTEGER,
version_2_1_2 INTEGER,
version_2_1_3 INTEGER,
version_2_2 INTEGER,
version_2_3 INTEGER,
version_2_3_0 INTEGER,
version_2_4_0 INTEGER,
version_2_5_0 INTEGER,
version_3_0 INTEGER,
version_3_0_1 INTEGER,
version_3_0_2 INTEGER,
version_3_0_5 INTEGER,
version_3_1 INTEGER,
version_3_1_1 INTEGER,
version_3_1_2 INTEGER,
version_3_2 INTEGER,
version_3_2_1 INTEGER,
version_3_2_2 INTEGER,
version_3_3 INTEGER,
version_3_3_1 INTEGER,
version_3_3_2 INTEGER,
version_3_4 INTEGER,
version_3_4_1 INTEGER,
version_3_4_2 INTEGER,
version_3_5 INTEGER,
version_3_5_1 INTEGER,
version_3_5_2 INTEGER,
version_3_6 INTEGER,
version_3_6_1 INTEGER,
version_3_6_2 INTEGER,
version_3_7 INTEGER,
version_4_0 INTEGER,
version_4_0_2 INTEGER,
version_4_0_3 INTEGER,
version_4_1 INTEGER,
version_4_2 INTEGER,
version_4_3 INTEGER,
version_4_4 INTEGER,
version_5_0 INTEGER,
version_5_0_1 INTEGER,
version_5_0_2 INTEGER,
version_6_0 INTEGER,
version_6_0_1 INTEGER,
version_6_0_2 INTEGER,
version_6_1 INTEGER,
version_7_0 INTEGER,
version_7_0_1 INTEGER,
version_7_0_2 INTEGER,
version_8_0 INTEGER,
version_DD_1_1 INTEGER,
version_DEVELOPMEN INTEGER,
op_sys_AIX_Motif INTEGER,
op_sys_All INTEGER,
op_sys_HP_UX INTEGER,
op_sys_Linux INTEGER,
op_sys_Linux_Qt INTEGER,
op_sys_Linux_GTK INTEGER,
op_sys_Linux_Motif INTEGER,
op_sys_Mac_OS_X INTEGER,
op_sys_Mac_OS_X_Cocoa INTEGER,
op_sys_MacOS_X INTEGER,
op_sys_Neutrino INTEGER,
op_sys_QNX_Photon INTEGER,
op_sys_Solaris INTEGER,
op_sys_Solaris_GTK INTEGER,
op_sys_Solaris_Motif INTEGER,
op_sys_SymbianOS_S60 INTEGER,
op_sys_Unix_All INTEGER,
op_sys_Windows_2003_Server INTEGER,
op_sys_Windows_7 INTEGER,
op_sys_Windows_95 INTEGER,
op_sys_Windows_98 INTEGER,
op_sys_Windows_All INTEGER,
op_sys_Windows_CE INTEGER,
op_sys_Windows_Mobile_2003 INTEGER,
op_sys_Windows_Mobile_5_0 INTEGER,
op_sys_Windows_NT INTEGER,
op_sys_Windows_Server_2003 INTEGER,
op_sys_Windows_Server_2008 INTEGER,
op_sys_Windows_Vista INTEGER,
op_sys_Windows_Vista_Beta_2 INTEGER,
op_sys_Windows_Vista_WPF INTEGER,
op_sys_Windows_XP INTEGER,
component_API_Tools INTEGER,
component_APT INTEGER,
component_Ant INTEGER,
component_Resources INTEGER,
component_Runtime INTEGER,
component_SWT INTEGER,
component_Scripting INTEGER,
component_Search INTEGER,
component_Team INTEGER,
component_Text INTEGER,
component_User_Assistance INTEGER,
component_VCM INTEGER,
component_WebDAV INTEGER,
component_Website INTEGER,
component_N INTEGER,
component_releng INTEGER,
component_ui INTEGER,
severity_blocker INTEGER,
severity_critical INTEGER,
severity_enhancement INTEGER,
severity_major INTEGER,
severity_minor INTEGER,
severity_normal INTEGER,
time_open REAL,
Edits_Influence REAL,
Avg_Success_Rate REAL,
Reporter_Success_Rate REAL,
Reporter_Reputation REAL,
Assignees_Avg_SR REAL
);

-- Create the Test table
CREATE TABLE Test (
who INTEGER,
current_resolution_FIXED INTEGER,
current_status_RESOLVED INTEGER,
current_status_VERIFIED INTEGER,
priority_P1 INTEGER,
priority_P2 INTEGER,
priority_P3 INTEGER,
priority_P4 INTEGER,
priority_P5 INTEGER,
product_CDT INTEGER,
product_JDT INTEGER,
product_Platform INTEGER,
version_0_DD_1_0 INTEGER,
version_0_DD_1_1 INTEGER,
version_0_1_3 INTEGER,
version_0_5 INTEGER,
version_0_7 INTEGER,
version_0_7_1 INTEGER,
version_0_8 INTEGER,
version_0_9 INTEGER,
version_0_9_2 INTEGER,
version_1_0 INTEGER,
version_1_0_0 INTEGER,
version_1_0_1 INTEGER,
version_1_0_2 INTEGER,
version_1_0_3 INTEGER,
version_1_1 INTEGER,
version_1_2 INTEGER,
version_1_2_1 INTEGER,
version_1_3_0 INTEGER,
version_1_3_1 INTEGER,
version_1_5 INTEGER,
version_1_5_2 INTEGER,
version_1_5_3 INTEGER,
version_1_7 INTEGER,
version_2_0 INTEGER,
version_2_0_0 INTEGER,
version_2_0_1 INTEGER,
version_2_0_2 INTEGER,
version_2_0_3 INTEGER,
version_2_1 INTEGER,
version_2_1_0 INTEGER,
version_2_1_1 INTEGER,
version_2_1_2 INTEGER,
version_2_1_3 INTEGER,
version_2_2 INTEGER,
version_2_3 INTEGER,
version_2_3_0 INTEGER,
version_2_4_0 INTEGER,
version_2_5_0 INTEGER,
version_3_0 INTEGER,
version_3_0_1 INTEGER,
version_3_0_2 INTEGER,
version_3_0_5 INTEGER,
version_3_1 INTEGER,
version_3_1_1 INTEGER,
version_3_1_2 INTEGER,
version_3_2 INTEGER,
version_3_2_1 INTEGER,
version_3_2_2 INTEGER,
version_3_3 INTEGER,
version_3_3_1 INTEGER,
version_3_3_2 INTEGER,
version_3_4 INTEGER,
version_3_4_1 INTEGER,
version_3_4_2 INTEGER,
version_3_5 INTEGER,
version_3_5_1 INTEGER,
version_3_5_2 INTEGER,
version_3_6 INTEGER,
version_3_6_1 INTEGER,
version_3_6_2 INTEGER,
version_3_7 INTEGER,
version_4_0 INTEGER,
version_4_0_2 INTEGER,
version_4_0_3 INTEGER,
version_4_1 INTEGER,
version_4_2 INTEGER,
version_4_3 INTEGER,
version_4_4 INTEGER,
version_5_0 INTEGER,
version_5_0_1 INTEGER,
version_5_0_2 INTEGER,
version_6_0 INTEGER,
version_6_0_1 INTEGER,
version_6_0_2 INTEGER,
version_6_1 INTEGER,
version_7_0 INTEGER,
version_7_0_1 INTEGER,
version_7_0_2 INTEGER,
version_8_0 INTEGER,
version_DD_1_1 INTEGER,
version_DEVELOPMEN INTEGER,
op_sys_AIX_Motif INTEGER,
op_sys_All INTEGER,
op_sys_HP_UX INTEGER,
op_sys_Linux INTEGER,
op_sys_Linux_Qt INTEGER,
op_sys_Linux_GTK INTEGER,
op_sys_Linux_Motif INTEGER,
op_sys_Mac_OS_X INTEGER,
op_sys_Mac_OS_X_Cocoa INTEGER,
op_sys_MacOS_X INTEGER,
op_sys_Neutrino INTEGER,
op_sys_QNX_Photon INTEGER,
op_sys_Solaris INTEGER,
op_sys_Solaris_GTK INTEGER,
op_sys_Solaris_Motif INTEGER,
op_sys_SymbianOS_S60 INTEGER,
op_sys_Unix_All INTEGER,
op_sys_Windows_2003_Server INTEGER,
op_sys_Windows_7 INTEGER,
op_sys_Windows_95 INTEGER,
op_sys_Windows_98 INTEGER,
op_sys_Windows_All INTEGER,
op_sys_Windows_CE INTEGER,
op_sys_Windows_Mobile_2003 INTEGER,
op_sys_Windows_Mobile_5_0 INTEGER,
op_sys_Windows_NT INTEGER,
op_sys_Windows_Server_2003 INTEGER,
op_sys_Windows_Server_2008 INTEGER,
op_sys_Windows_Vista INTEGER,
op_sys_Windows_Vista_Beta_2 INTEGER,
op_sys_Windows_Vista_WPF INTEGER,
op_sys_Windows_XP INTEGER,
component_API_Tools INTEGER,
component_APT INTEGER,
component_Ant INTEGER,
component_Resources INTEGER,
component_Runtime INTEGER,
component_SWT INTEGER,
component_Scripting INTEGER,
component_Search INTEGER,
component_Team INTEGER,
component_Text INTEGER,
component_User_Assistance INTEGER,
component_VCM INTEGER,
component_WebDAV INTEGER,
component_Website INTEGER,
component_N INTEGER,
component_releng INTEGER,
component_ui INTEGER,
severity_blocker INTEGER,
severity_critical INTEGER,
severity_enhancement INTEGER,
severity_major INTEGER,
severity_minor INTEGER,
severity_normal INTEGER,
time_open REAL,
Edits_Influence REAL,
Avg_Success_Rate REAL,
Reporter_Success_Rate REAL,
Reporter_Reputation REAL,
Assignees_Avg_SR REAL
);

-- Create the Validation table
CREATE TABLE Validation (
who INTEGER,
current_resolution_FIXED INTEGER,
current_status_RESOLVED INTEGER,
current_status_VERIFIED INTEGER,
priority_P1 INTEGER,
priority_P2 INTEGER,
priority_P3 INTEGER,
priority_P4 INTEGER,
priority_P5 INTEGER,
product_CDT INTEGER,
product_JDT INTEGER,
product_Platform INTEGER,
version_0_DD_1_0 INTEGER,
version_0_DD_1_1 INTEGER,
version_0_1_3 INTEGER,
version_0_5 INTEGER,
version_0_7 INTEGER,
version_0_7_1 INTEGER,
version_0_8 INTEGER,
version_0_9 INTEGER,
version_0_9_2 INTEGER,
version_1_0 INTEGER,
version_1_0_0 INTEGER,
version_1_0_1 INTEGER,
version_1_0_2 INTEGER,
version_1_0_3 INTEGER,
version_1_1 INTEGER,
version_1_2 INTEGER,
version_1_2_1 INTEGER,
version_1_3_0 INTEGER,
version_1_3_1 INTEGER,
version_1_5 INTEGER,
version_1_5_2 INTEGER,
version_1_5_3 INTEGER,
version_1_7 INTEGER,
version_2_0 INTEGER,
version_2_0_0 INTEGER,
version_2_0_1 INTEGER,
version_2_0_2 INTEGER,
version_2_0_3 INTEGER,
version_2_1 INTEGER,
version_2_1_0 INTEGER,
version_2_1_1 INTEGER,
version_2_1_2 INTEGER,
version_2_1_3 INTEGER,
version_2_2 INTEGER,
version_2_3 INTEGER,
version_2_3_0 INTEGER,
version_2_4_0 INTEGER,
version_2_5_0 INTEGER,
version_3_0 INTEGER,
version_3_0_1 INTEGER,
version_3_0_2 INTEGER,
version_3_0_5 INTEGER,
version_3_1 INTEGER,
version_3_1_1 INTEGER,
version_3_1_2 INTEGER,
version_3_2 INTEGER,
version_3_2_1 INTEGER,
version_3_2_2 INTEGER,
version_3_3 INTEGER,
version_3_3_1 INTEGER,
version_3_3_2 INTEGER,
version_3_4 INTEGER,
version_3_4_1 INTEGER,
version_3_4_2 INTEGER,
version_3_5 INTEGER,
version_3_5_1 INTEGER,
version_3_5_2 INTEGER,
version_3_6 INTEGER,
version_3_6_1 INTEGER,
version_3_6_2 INTEGER,
version_3_7 INTEGER,
version_4_0 INTEGER,
version_4_0_2 INTEGER,
version_4_0_3 INTEGER,
version_4_1 INTEGER,
version_4_2 INTEGER,
version_4_3 INTEGER,
version_4_4 INTEGER,
version_5_0 INTEGER,
version_5_0_1 INTEGER,
version_5_0_2 INTEGER,
version_6_0 INTEGER,
version_6_0_1 INTEGER,
version_6_0_2 INTEGER,
version_6_1 INTEGER,
version_7_0 INTEGER,
version_7_0_1 INTEGER,
version_7_0_2 INTEGER,
version_8_0 INTEGER,
version_DD_1_1 INTEGER,
version_DEVELOPMEN INTEGER,
op_sys_AIX_Motif INTEGER,
op_sys_All INTEGER,
op_sys_HP_UX INTEGER,
op_sys_Linux INTEGER,
op_sys_Linux_Qt INTEGER,
op_sys_Linux_GTK INTEGER,
op_sys_Linux_Motif INTEGER,
op_sys_Mac_OS_X INTEGER,
op_sys_Mac_OS_X_Cocoa INTEGER,
op_sys_MacOS_X INTEGER,
op_sys_Neutrino INTEGER,
op_sys_QNX_Photon INTEGER,
op_sys_Solaris INTEGER,
op_sys_Solaris_GTK INTEGER,
op_sys_Solaris_Motif INTEGER,
op_sys_SymbianOS_S60 INTEGER,
op_sys_Unix_All INTEGER,
op_sys_Windows_2003_Server INTEGER,
op_sys_Windows_7 INTEGER,
op_sys_Windows_95 INTEGER,
op_sys_Windows_98 INTEGER,
op_sys_Windows_All INTEGER,
op_sys_Windows_CE INTEGER,
op_sys_Windows_Mobile_2003 INTEGER,
op_sys_Windows_Mobile_5_0 INTEGER,
op_sys_Windows_NT INTEGER,
op_sys_Windows_Server_2003 INTEGER,
op_sys_Windows_Server_2008 INTEGER,
op_sys_Windows_Vista INTEGER,
op_sys_Windows_Vista_Beta_2 INTEGER,
op_sys_Windows_Vista_WPF INTEGER,
op_sys_Windows_XP INTEGER,
component_API_Tools INTEGER,
component_APT INTEGER,
component_Ant INTEGER,
component_Resources INTEGER,
component_Runtime INTEGER,
component_SWT INTEGER,
component_Scripting INTEGER,
component_Search INTEGER,
component_Team INTEGER,
component_Text INTEGER,
component_User_Assistance INTEGER,
component_VCM INTEGER,
component_WebDAV INTEGER,
component_Website INTEGER,
component_N INTEGER,
component_releng INTEGER,
component_ui INTEGER,
severity_blocker INTEGER,
severity_critical INTEGER,
severity_enhancement INTEGER,
severity_major INTEGER,
severity_minor INTEGER,
severity_normal INTEGER,
time_open REAL,
Edits_Influence REAL,
Avg_Success_Rate REAL,
Reporter_Success_Rate REAL,
Reporter_Reputation REAL,
Assignees_Avg_SR REAL
);