add_executable(test_lstm_forward test_lstm_forward.cpp)

add_test(NAME LSTM_Forward_Parity_Test COMMAND test_lstm_forward)

set_tests_properties(LSTM_Forward_Parity_Test PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)