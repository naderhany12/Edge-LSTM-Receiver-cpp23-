# 1. Forward Parity Test
add_executable(test_lstm_forward test_lstm_forward.cpp)

add_test(NAME LSTM_Forward_Parity_Test COMMAND test_lstm_forward)

set_tests_properties(LSTM_Forward_Parity_Test PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# 2. On-Device Adaptation Test
add_executable(test_on_device_adaptation test_on_device_adaptation.cpp)

add_test(NAME On_Device_Adaptation_Test COMMAND test_on_device_adaptation)

set_tests_properties(On_Device_Adaptation_Test PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)